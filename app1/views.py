import os
from tokenize import String

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from .models import Student, Attendance, CameraConfiguration, Bus, DriverLocation
from django.core.files.base import ContentFile
from datetime import datetime, timedelta
from django.utils import timezone
import pygame  # Import pygame for playing sounds
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.urls import reverse_lazy
from django.contrib.auth.decorators import login_required
import threading
import time
import base64
from django.db import IntegrityError
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from .models import Student
from django.shortcuts import render
from django.http import HttpResponse
import csv
from django.http import JsonResponse
import json
from django.views.decorators.http import require_POST
import pytz
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_http_methods
from django.db import transaction


# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Function to detect and encode faces
def detect_and_encode(image):
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            faces = []
            for box in boxes:
                face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0)
                encoding = resnet(face_tensor).detach().numpy().flatten()
                faces.append(encoding)
            return faces
    return []

# Function to encode uploaded images
def encode_uploaded_images():
    known_face_encodings = []
    known_face_names = []

    # Fetch only authorized images
    uploaded_images = Student.objects.filter(authorized=True)

    for student in uploaded_images:
        # Fix 1: Get the complete file name, not just the directory
        if not student.image:  # Add check for empty image field
            print(f"No image file found for student {student.name}")
            continue
            
        image_path = os.path.join(settings.MEDIA_ROOT, str(student.image))
        
        # Debug information
        print(f"Attempting to read image for {student.name}")
        print(f"Full image path: {image_path}")
        print(f"File exists: {os.path.exists(image_path)}")
        
        if not os.path.exists(image_path):
            print(f"Image file does not exist for student {student.name}")
            continue

        try:  # Add error handling
            known_image = cv2.imread(image_path)
            if known_image is None:
                print(f"Failed to read image for student {student.name}")
                continue
                
            if known_image.size == 0:
                print(f"Empty image for student {student.name}")
                continue

            known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
            encodings = detect_and_encode(known_image_rgb)
            if encodings:
                known_face_encodings.extend(encodings)
                known_face_names.append(student.name)
        except Exception as e:
            print(f"Error processing image for student {student.name}: {str(e)}")
            continue

    return known_face_encodings, known_face_names

# Function to recognize faces
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized_names = []
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < threshold:
            recognized_names.append(known_names[min_distance_idx])
        else:
            recognized_names.append('Not Recognized')
    return recognized_names

# View for capturing student information and image
@login_required
def capture_student(request):
    if request.method == 'POST':
        try:
            name = request.POST.get('name')
            email = request.POST.get('email')
            phone = request.POST.get('phone')
            student_class = request.POST.get('student_class')
            authorized = request.POST.get('authorized') == 'on'

            # Handle image upload
            image = None
            if 'student_image' in request.FILES:
                image = request.FILES['student_image']
            elif 'captured_image' in request.POST and request.POST['captured_image']:
                # Handle captured image from camera
                image_data = request.POST['captured_image']
                if image_data.startswith('data:image'):
                    format, imgstr = image_data.split(';base64,')
                    ext = format.split('/')[-1]
                    image = ContentFile(
                        base64.b64decode(imgstr), 
                        name=f'student_capture_{timezone.now().timestamp()}.{ext}'
                    )

            # Create student
            student = Student.objects.create(
                name=name,
                email=email,
                phone_number=phone,
                student_class=student_class,
                image=image,
                authorized=authorized
            )

            return JsonResponse({'success': True})

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return render(request, 'capture_student.html')


# Success view after capturing student information and image
def selfie_success(request):
    return render(request, 'selfie_success.html')


# This views for capturing studen faces and recognize
def capture_and_recognize(request):
    stop_events = []  # List to store stop events for each thread
    camera_threads = []  # List to store threads for each camera
    camera_windows = []  # List to store window names
    error_messages = []  # List to capture errors from threads

    def process_frame(cam_config, stop_event):
        """Thread function to capture and process frames for each camera."""
        cap = None
        window_created = False  # Flag to track if the window was created
        try:
            # Check if the camera source is a number (local webcam) or a string (IP camera URL)
            if cam_config.camera_source.isdigit():
                cap = cv2.VideoCapture(int(cam_config.camera_source))  # Use integer index for webcam
            else:
                cap = cv2.VideoCapture(cam_config.camera_source)  # Use string for IP camera URL

            if not cap.isOpened():
                raise Exception(f"Unable to access camera {cam_config.name}.")

            threshold = cam_config.threshold

            # Initialize pygame mixer for sound playback
            pygame.mixer.init()
            success_sound = pygame.mixer.Sound('app1/suc.wav')  # load sound path

            window_name = f'Face Recognition - {cam_config.name}'
            camera_windows.append(window_name)  # Track the window name

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to capture frame for camera: {cam_config.name}")
                    break  # If frame capture fails, break from the loop

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                test_face_encodings = detect_and_encode(frame_rgb)  # Function to detect and encode face in frame

                if test_face_encodings:
                    known_face_encodings, known_face_names = encode_uploaded_images()  # Load known face encodings once
                    if known_face_encodings:
                        names = recognize_faces(np.array(known_face_encodings), known_face_names, test_face_encodings, threshold)

                        for name, box in zip(names, mtcnn.detect(frame_rgb)[0]):
                            if box is not None:
                                (x1, y1, x2, y2) = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                                if name != 'Not Recognized':
                                    students = Student.objects.filter(name=name)
                                    if students.exists():
                                        student = students.first()
                                        current_time = timezone.now()
                                        current_date = current_time.date()

                                        # Get or create today's attendance record
                                        attendance = Attendance.objects.filter(
                                            student=student,
                                            date=current_date
                                        ).order_by('-check_in_time').first()

                                        if not attendance:
                                            # First check-in of the day
                                            attendance = Attendance.objects.create(
                                                student=student,
                                                date=current_date,
                                                check_in_time=current_time
                                            )
                                            student.is_on_bus = True
                                            student.save()
                                            success_sound.play()
                                            cv2.putText(frame, f"{name} checked in.", (50, 50), 
                                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                                        elif attendance.check_in_time and not attendance.check_out_time:
                                            # Check if 1 minute has passed since check-in
                                            time_since_checkin = current_time - attendance.check_in_time
                                            if time_since_checkin >= timedelta(minutes=1):
                                                attendance.check_out_time = current_time
                                                attendance.save()
                                                student.is_on_bus = False
                                                student.save()
                                                success_sound.play()
                                                cv2.putText(frame, f"{name} checked out.", (50, 50), 
                                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                            else:
                                                remaining_seconds = 60 - time_since_checkin.seconds
                                                cv2.putText(frame, f"Wait {remaining_seconds}s to check out", 
                                                          (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                                        elif attendance.check_out_time:
                                            # Check if 2 minutes have passed since last check-out
                                            time_since_checkout = current_time - attendance.check_out_time
                                            if time_since_checkout >= timedelta(minutes=2):
                                                # Create new attendance record for new check-in
                                                attendance = Attendance.objects.create(
                                                    student=student,
                                                    date=current_date,
                                                    check_in_time=current_time
                                                )
                                                student.is_on_bus = True
                                                student.save()
                                                success_sound.play()
                                                cv2.putText(frame, f"{name} checked in.", (50, 50), 
                                                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                            else:
                                                remaining_seconds = 120 - time_since_checkout.seconds
                                                cv2.putText(frame, f"Wait {remaining_seconds}s to check in", 
                                                          (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Display frame in separate window for each camera
                if not window_created:
                    cv2.namedWindow(window_name)  # Only create window once
                    window_created = True  # Mark window as created
                
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()  # Signal the thread to stop when 'q' is pressed
                    break

        except Exception as e:
            print(f"Error in thread for {cam_config.name}: {e}")
            error_messages.append(str(e))  # Capture error message
        finally:
            if cap is not None:
                cap.release()
            if window_created:
                cv2.destroyWindow(window_name)  # Only destroy if window was created

    try:
        # Get all camera configurations
        cam_configs = CameraConfiguration.objects.all()
        if not cam_configs.exists():
            raise Exception("No camera configurations found. Please configure them in the admin panel.")

        # Create threads for each camera configuration
        for cam_config in cam_configs:
            stop_event = threading.Event()
            stop_events.append(stop_event)

            camera_thread = threading.Thread(target=process_frame, args=(cam_config, stop_event))
            camera_threads.append(camera_thread)
            camera_thread.start()

        # Keep the main thread running while cameras are being processed
        while any(thread.is_alive() for thread in camera_threads):
            time.sleep(1)  # Non-blocking wait, allowing for UI responsiveness

    except Exception as e:
        error_messages.append(str(e))  # Capture the error message
    finally:
        # Ensure all threads are signaled to stop
        for stop_event in stop_events:
            stop_event.set()

        # Ensure all windows are closed in the main thread
        for window in camera_windows:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:  # Check if window exists
                cv2.destroyWindow(window)

    # Check if there are any error messages
    if error_messages:
        # Join all error messages into a single string
        full_error_message = "\n".join(error_messages)
        return render(request, 'error.html', {'error_message': full_error_message})  # Render the error page with message

    return redirect('student_attendance_list')

#this is for showing Attendance list
def student_attendance_list(request):
    search_query = request.GET.get('search', '')
    date_filter = request.GET.get('attendance_date', '')
    bus_filter = request.GET.get('bus', '')
    
    # Get all students
    students = Student.objects.all()
    
    # Apply search filter
    if search_query:
        students = students.filter(name__icontains=search_query)
    
    # Apply bus filter
    if bus_filter:
        students = students.filter(bus_id=bus_filter)
    
    # Get all buses for the dropdown
    buses = Bus.objects.all()
    
    student_attendance_data = []
    for student in students:
        attendance_records = Attendance.objects.filter(student=student)
        
        # Apply date filter
        if date_filter:
            attendance_records = attendance_records.filter(date=date_filter)
            
        if attendance_records.exists():
            student_attendance_data.append({
                'student': student,
                'attendance_records': attendance_records
            })
    
    # Handle CSV export
    if request.GET.get('export') == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="attendance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
        
        writer = csv.writer(response)
        # Write CSV header
        writer.writerow([
            'Student ID',
            'Student Name',
            'Class',
            'Date',
            'Check-in Time',
            'Check-out Time',
            'Duration'
        ])
        
        # Write data rows
        for data in student_attendance_data:
            student = data['student']
            for attendance in data['attendance_records']:
                # Safely handle check-in and check-out times
                check_in_time = attendance.check_in_time.strftime('%H:%M:%S') if attendance.check_in_time else 'Not Checked In'
                check_out_time = attendance.check_out_time.strftime('%H:%M:%S') if attendance.check_out_time else 'Not Checked Out'
                
                writer.writerow([
                    student.id,
                    student.name,
                    student.student_class,
                    attendance.date.strftime('%Y-%m-%d'),
                    check_in_time,
                    check_out_time,
                    attendance.calculate_duration() if attendance.check_in_time and attendance.check_out_time else "Not Complete"
                ])
        
        return response

    context = {
        'student_attendance_data': student_attendance_data,
        'search_query': search_query,
        'date_filter': date_filter,
        'bus_filter': bus_filter,
        'buses': buses,
    }
    return render(request, 'student_attendance_list.html', context)


def home(request):
    return render(request, 'home.html')




# Custom user pass test for admin access
def is_admin(user):
    return user.is_superuser

@login_required
@user_passes_test(is_admin)
def student_list(request):
    students = Student.objects.all()
    buses = Bus.objects.all()
    selected_bus = request.GET.get('bus')
    
    if selected_bus:
        try:
            selected_bus = int(selected_bus)  # Convert to integer
            students = students.filter(bus_id=selected_bus)
            available_students = Student.objects.filter(bus__isnull=True)
        except ValueError:
            selected_bus = None
            available_students = []
    else:
        selected_bus = None
        available_students = []
        
    context = {
        'students': students,
        'buses': buses,
        'selected_bus': selected_bus,
        'available_students': available_students
    }
    return render(request, 'student_list.html', context)

@login_required
@user_passes_test(is_admin)
def student_detail(request, pk):
    student = get_object_or_404(Student, pk=pk)
    return render(request, 'student_detail.html', {'student': student})

@login_required
@user_passes_test(is_admin)
def student_authorize(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    if request.method == 'POST':
        authorized = request.POST.get('authorized', False)
        student.authorized = bool(authorized)
        student.save()
        return redirect('student-detail', pk=pk)
    
    return render(request, 'student_authorize.html', {'student': student})

# This views is for Deleting student
@login_required
@user_passes_test(is_admin)
def student_delete(request, pk):
    student = get_object_or_404(Student, pk=pk)
    
    if request.method == 'POST':
        student.delete()
        messages.success(request, 'Student deleted successfully.')
        return redirect('student-list')  # Redirect to the student list after deletion
    
    return render(request, 'student_delete_confirm.html', {'student': student})


# View function for user login
def user_login(request):
    # Check if the request method is POST, indicating a form submission
    if request.method == 'POST':
        # Retrieve username and password from the submitted form data
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Authenticate the user using the provided credentials
        user = authenticate(request, username=username, password=password)

        # Check if the user was successfully authenticated
        if user is not None:
            # Log the user in by creating a session
            login(request, user)
            # Redirect the user to the student list page after successful login
            return redirect('home')  # Replace 'student-list' with your desired redirect URL after login
        else:
            # If authentication fails, display an error message
            messages.error(request, 'Invalid username or password.')

    # Render the login template for GET requests or if authentication fails
    return render(request, 'login.html')


# This is for user logout
def user_logout(request):
    logout(request)
    return redirect('login')  # Replace 'login' with your desired redirect URL after logout

# Function to handle the creation of a new camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_create(request):
    # Check if the request method is POST, indicating form submission
    if request.method == "POST":
        # Retrieve form data from the request
        name = request.POST.get('name')
        camera_source = request.POST.get('camera_source')
        threshold = request.POST.get('threshold')

        try:
            # Save the data to the database using the CameraConfiguration model
            CameraConfiguration.objects.create(
                name=name,
                camera_source=camera_source,
                threshold=threshold,
            )
            # Redirect to the list of camera configurations after successful creation
            return redirect('camera_config_list')

        except IntegrityError:
            # Handle the case where a configuration with the same name already exists
            messages.error(request, "A configuration with this name already exists.")
            # Render the form again to allow user to correct the error
            return render(request, 'camera_config_form.html')

    # Render the camera configuration form for GET requests
    return render(request, 'camera_config_form.html')


# READ: Function to list all camera configurations
@login_required
@user_passes_test(is_admin)
def camera_config_list(request):
    # Retrieve all CameraConfiguration objects from the database
    configs = CameraConfiguration.objects.all()
    # Render the list template with the retrieved configurations
    return render(request, 'camera_config_list.html', {'configs': configs})


# UPDATE: Function to edit an existing camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_update(request, pk):
    # Retrieve the specific configuration by primary key or return a 404 error if not found
    config = get_object_or_404(CameraConfiguration, pk=pk)

    # Check if the request method is POST, indicating form submission
    if request.method == "POST":
        # Update the configuration fields with data from the form
        config.name = request.POST.get('name')
        config.camera_source = request.POST.get('camera_source')
        config.threshold = request.POST.get('threshold')
        config.success_sound_path = request.POST.get('success_sound_path')

        # Save the changes to the database
        config.save()  

        # Redirect to the list page after successful update
        return redirect('camera_config_list')  
    
    # Render the configuration form with the current configuration data for GET requests
    return render(request, 'camera_config_form.html', {'config': config})


# DELETE: Function to delete a camera configuration
@login_required
@user_passes_test(is_admin)
def camera_config_delete(request, pk):
    # Retrieve the specific configuration by primary key or return a 404 error if not found
    config = get_object_or_404(CameraConfiguration, pk=pk)

    # Check if the request method is POST, indicating confirmation of deletion
    if request.method == "POST":
        # Delete the record from the database
        config.delete()  
        # Redirect to the list of camera configurations after deletion
        return redirect('camera_config_list')

    # Render the delete confirmation template with the configuration data
    return render(request, 'camera_config_delete.html', {'config': config})

def bus_attendance(request):
    buses = Bus.objects.all()
    students = Student.objects.all()
    context = {
        'buses': buses,
        'students': students
    }
    return render(request, 'bus_attendance.html', context)

@login_required
@user_passes_test(is_admin)
def add_student_to_bus(request, bus_id):
    if request.method == 'POST':
        student_id = request.POST.get('student_id')
        bus = get_object_or_404(Bus, id=bus_id)
        student = get_object_or_404(Student, id=student_id)
        student.bus = bus
        student.save()
        return redirect('student-list')
    return redirect('student-list')

@login_required
@user_passes_test(is_admin)
def bus_list(request):
    buses = Bus.objects.all()
    return render(request, 'bus_list.html', {'buses': buses})

@login_required
@user_passes_test(is_admin)
def bus_create(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        route = request.POST.get('route')
        capacity = request.POST.get('capacity')
        driver_name = request.POST.get('driver_name')
        driver_phone = request.POST.get('driver_phone')
        
        bus = Bus.objects.create(
            name=name,
            route=route,
            capacity=capacity,
            driver_name=driver_name,
            driver_phone=driver_phone
        )
        return redirect('bus-list')
    
    return render(request, 'bus_form.html')

@login_required
@user_passes_test(is_admin)
def bus_edit(request, pk):
    bus = get_object_or_404(Bus, pk=pk)
    
    if request.method == 'POST':
        bus.name = request.POST.get('name')
        bus.route = request.POST.get('route')
        bus.capacity = request.POST.get('capacity')
        bus.driver_name = request.POST.get('driver_name')
        bus.driver_phone = request.POST.get('driver_phone')
        bus.save()
        return redirect('bus-list')
    
    return render(request, 'bus_form.html', {'bus': bus})

@login_required
@user_passes_test(is_admin)
def bus_delete(request, pk):
    bus = get_object_or_404(Bus, pk=pk)
    if request.method == 'POST':
        bus.delete()
        return redirect('bus-list')
    return render(request, 'bus_delete_confirm.html', {'bus': bus})

def toggle_student_bus_status(request, student_id, bus_id):
    if request.method == 'POST':
        student = get_object_or_404(Student, id=student_id)
        bus = get_object_or_404(Bus, id=bus_id)
        current_time = timezone.now()
        current_date = current_time.date()

        # Get or create today's attendance record
        attendance, created = Attendance.objects.get_or_create(
            student=student,
            date=current_date
        )

        # If student is not on bus (last action was check-out or no action)
        if not student.is_on_bus:
            # Clear any previous check-out time and set new check-in time
            attendance.check_out_time = None
            attendance.check_in_time = current_time
            student.is_on_bus = True
        else:
            # Clear any previous check-in time and set new check-out time
            attendance.check_in_time = None
            attendance.check_out_time = current_time
            student.is_on_bus = False

        attendance.save()
        student.save()

        return redirect('bus-detail', pk=bus_id)

    return redirect('bus-detail', pk=bus_id)

def register_student(request):
    if request.method == 'POST':
        # Your existing form validation code...

        # Handle image upload
        image = None
        if 'student_image' in request.FILES:
            image = request.FILES['student_image']
        elif 'captured_image' in request.POST and request.POST['captured_image']:
            # Handle captured image from camera
            image_data = request.POST['captured_image']
            if image_data.startswith('data:image'):
                # Remove the data URL prefix
                format, imgstr = image_data.split(';base64,')
                ext = format.split('/')[-1]
                
                # Convert base64 to file
                image = ContentFile(
                    base64.b64decode(imgstr), 
                    name=f'student_capture_{timezone.now().timestamp()}.{ext}'
                )

        # Create student with image
        student = Student.objects.create(
            name=name,
            email=email,
            phone_number=phone,
            student_class=student_class,
            image=image,
            authorized=False
        )

        return redirect('registration-success')

    return render(request, 'capture_student.html')

def bus_detail(request, pk):
    bus = get_object_or_404(Bus, pk=pk)
    students = Student.objects.filter(bus=bus).order_by('name')
    current_date = timezone.now().date()
    
    for student in students:
        # Get today's latest attendance record
        attendance = Attendance.objects.filter(
            student=student,
            date=current_date
        ).order_by('-check_in_time').first()
        
        if attendance:
            if attendance.check_in_time and not attendance.check_out_time:
                # Has checked in but not checked out
                student.current_status = 'on_bus'
                student.last_action_time = attendance.check_in_time
            elif attendance.check_out_time:
                # Has checked out
                student.current_status = 'checked_out'
                student.last_action_time = attendance.check_out_time
            else:
                # No check-in or check-out today
                student.current_status = 'not_on_bus'
                student.last_action_time = None
        else:
            # No attendance record today
            student.current_status = 'not_on_bus'
            student.last_action_time = None

    context = {
        'bus': bus,
        'students': students,
        'student_count': students.count()
    }
    return render(request, 'bus_detail.html', context)

def driver_tracking(request, bus_id):
    bus = get_object_or_404(Bus, id=bus_id)
    students = Student.objects.filter(bus=bus).select_related('studentaddress')
    
    context = {
        'bus': bus,
        'students': students,
    }
    return render(request, 'driver_tracking.html', context)

def bus_tracking_list(request):
    buses = Bus.objects.all().order_by('name')
    
    # Get current location for each bus
    for bus in buses:
        try:
            location = DriverLocation.objects.get(bus=bus)
            bus.has_location = True
            bus.last_updated = location.last_updated
        except DriverLocation.DoesNotExist:
            bus.has_location = False
            bus.last_updated = None
    
    context = {
        'buses': buses
    }
    return render(request, 'bus_tracking_list.html', context)

@login_required
def driver_dashboard(request, bus_id=None):
    if bus_id is None:
        buses = Bus.objects.all()
        return render(request, 'driver_dashboard_select.html', {'buses': buses})
    
    bus = get_object_or_404(Bus, id=bus_id)
    # Remove select_related since student_class is not a relation
    students = Student.objects.filter(bus=bus)
    
    context = {
        'bus': bus,
        'students': students,
    }
    return render(request, 'driver_dashboard.html', context)

@ensure_csrf_cookie
@require_http_methods(["POST"])
def update_driver_location(request, bus_id):
    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            with transaction.atomic():
                bus = Bus.objects.get(id=bus_id)
                data = json.loads(request.body)
                
                latitude = float(data.get('latitude', 0))
                longitude = float(data.get('longitude', 0))
                
                if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
                    return JsonResponse({
                        'status': 'error',
                        'message': 'Invalid coordinates'
                    }, status=400)

                location, created = DriverLocation.objects.update_or_create(
                    bus=bus,
                    defaults={
                        'latitude': latitude,
                        'longitude': longitude,
                        'last_updated': timezone.now()
                    }
                )
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'Location updated successfully'
                })

        except Bus.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': 'Bus not found'
            }, status=404)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return JsonResponse({
                'status': 'error',
                'message': f'Database error: {str(e)}'
            }, status=500)

def get_bus_locations(request):
    buses = Bus.objects.filter(is_active=True)
    locations = []
    
    vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    
    for bus in buses:
        try:
            location = DriverLocation.objects.get(bus=bus)
            vietnam_time = location.last_updated.astimezone(vietnam_tz)
            
            locations.append({
                'bus_id': bus.id,
                'name': bus.name,
                'latitude': location.latitude,
                'longitude': location.longitude,
                'last_updated': vietnam_time.strftime('%H:%M:%S'),
                'driver_name': bus.driver_name,
                'route': bus.route
            })
        except DriverLocation.DoesNotExist:
            continue
    
    return JsonResponse({'locations': locations})

@require_http_methods(["POST"])
def update_bus_status(request, bus_id):
    try:
        bus = Bus.objects.get(id=bus_id)
        data = json.loads(request.body)
        is_active = data.get('is_active', False)
        
        if not is_active:
            # Clear the location when bus is turned off
            DriverLocation.objects.filter(bus=bus).delete()
        
        bus.is_active = is_active
        bus.save()
        
        return JsonResponse({
            'status': 'success',
            'message': 'Bus status updated successfully'
        })
    except Bus.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Bus not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@login_required
def get_student_status(request, bus_id):
    today = timezone.now().date()
    students = Student.objects.filter(bus_id=bus_id)
    student_data = []

    for student in students:
        attendance = Attendance.objects.filter(
            student=student,
            date=today
        ).first()

        data = {
            'id': student.id,
            'name': student.name,
            'student_class': student.student_class,
            'is_on_bus': student.is_on_bus,
            'last_check_out': attendance.check_out_time.isoformat() if attendance and attendance.check_out_time else None
        }

        student_data.append(data)

    return JsonResponse(student_data, safe=False)

@login_required
def get_student_detail(request, student_id):
    student = get_object_or_404(Student, id=student_id)
    attendance = Attendance.objects.filter(
        student=student,
        date=timezone.now().date()
    ).first()

    data = {
        'id': student.id,
        'name': student.name,
        'student_class': student.student_class,
        'phone_number': student.phone_number,
        'image': student.image.url if student.image else None,
        'is_on_bus': student.is_on_bus,
        'last_check_out': attendance.check_out_time.isoformat() if attendance and attendance.check_out_time else None
    }
    
    return JsonResponse(data)
