import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .models import Bus, DriverLocation, Student

class DriverLocationConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.bus_id = self.scope['url_route']['kwargs']['bus_id']
        self.room_group_name = f'bus_{self.bus_id}'

        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        data = json.loads(text_data)
        await self.update_location(data)
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'location_update',
                'data': data
            }
        )

    async def location_update(self, event):
        await self.send(text_data=json.dumps(event['data']))

    @database_sync_to_async
    def update_location(self, data):
        DriverLocation.objects.update_or_create(
            bus_id=self.bus_id,
            defaults={
                'latitude': data['latitude'],
                'longitude': data['longitude']
            }
        ) 