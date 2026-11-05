# SMART HOME LIGHTING SYSTEM
Home automation is becoming more and more common and advanced, with systems like amazon Alexa and smart plugs, we can control devices around our home on our personal devices, but what if it's even simpler? The concept of this project is simple. have your home devices actively respond to you, without having to make any control aside from initial setup. 

## THE IDEA
Every home is separated into different areas: hallway, living room, dining room, kitchen, etc. The idea is for the device in those areas turn on and off automatically whether there's a person present or not. 
<img width="847" height="801" alt="image" src="https://github.com/user-attachments/assets/a1ffc3a6-d8ec-4fde-b423-7f372adeea4d" />

To accomplish this, we need the following:

| Part | Function | 
| :--- | :---: |
| ARDU Wide angle IR camera | Detect and determine where the person is, works in complete darkness as it sees infrared light |
| RaspberryPi 5 with OS | The brain of the whole system, receive camera feed, send microcontroller signals | 
| IR led illuminator | Emit and bounce IR light around the room for the camera to pickup |
| Third Reality Zigbee smart plugs | Controls device on/off, works without wifi after setup|
| Sonoff ZBDongle-E | Makes the raspberrypi the "hub" of the whole system and control the plugs |
| 12V 3A power supply | power |
| Buck converter | The raspberrypi 5 runs on 5V but the IR led is 12V |
| IR-pass acrylic sheet | Hide the illuminator in the casing |
