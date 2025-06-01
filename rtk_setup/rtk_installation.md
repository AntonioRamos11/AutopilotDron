# RTK GPS Setup Guide for High-Precision Navigation

This guide outlines the process of setting up an RTK GPS system for high-precision drone navigation.

## Hardware Requirements

### Base Station
- u-blox ZED-F9P RTK GPS module
- Active GPS antenna with ground plane
- Tripod or fixed mounting point
- Power source (battery or mains with appropriate voltage regulator)
- Telemetry radio (for corrections transmission)

### Rover (Drone)
- u-blox ZED-F9P RTK GPS module
- Active GPS antenna (lightweight)
- Telemetry radio receiver

## Base Station Setup

1. **Mounting the Base Station**
   - Place the base station on a stable, elevated platform with clear sky vi
   ew
   - Ensure the antenna has an unobstructed view of the sky (minimum 15° above horizon)
   - For best results, use a survey-grade tripod or fixed mounting point
   - The base station should ideally be within 10 km of the drone's operating area

2. **Power and Connections**
   - Connect the GPS antenna to the ZED-F9P module
   - Connect power supply (typically 5V DC)
   - Connect the telemetry radio to the UART/UART2 port

3. **Base Station Configuration**
   - Use u-center software to configure the ZED-F9P:
     - Set RTCM3 message output on UART2 (1005, 1077, 1087, 1097, 1127, 1230)
     - Configure update rate to 5Hz or 10Hz
     - Set the base station to survey-in or fixed position mode
     - For survey-in, use minimum 60s duration and 2m accuracy

4. **Survey-In Procedure**
   - Allow the base station to complete a "survey-in" procedure to determine its exact position
   - Typical settings: 5 minutes minimum time, 2m maximum position variance
   - Once survey-in is complete, the position is stored and corrections will be transmitted

## Drone Setup

1. **Hardware Installation**
   - Mount the GPS antenna on top of the drone, away from EMI sources
   - Connect the GPS module to UART/I2C port on the flight controller
   - Connect the telemetry receiver to the GPS module

2. **PX4 Configuration**
   - Set the following parameters:
     - `GPS_1_CONFIG`: Select u-blox protocol
     - `GPS_1_GNSS`: Enable GPS, GLONASS, Galileo, BeiDou as available
     - `EKF2_GPS_CTRL`: 1 (enable GPS)
     - `EKF2_HGT_MODE`: 3 (use GPS for height estimation)
   - For dual GPS setups, configure `GPS_2_*` parameters similarly

3. **Testing RTK Fix**
   - In QGroundControl, verify that:
     - RTK status shows "RTK float" or "RTK fixed"
     - Horizontal position accuracy is < 0.1m (typically 0.01-0.03m in RTK fixed)
     - Satellite count is > 15 (ideally > 20)

## Common Issues and Troubleshooting

### No RTK Fix
- Ensure base station has clear sky view
- Check telemetry link signal strength
- Verify RTCM3 messages are being received by the drone
- Check GPS antenna placement and cable integrity

### Poor Accuracy
- Increase base station survey-in time
- Check for multipath interference near the base station
- Ensure the base station is stable and doesn't move during operation
- Consider environmental conditions (ionospheric activity, etc.)

### Lost RTK Fix During Flight
- Increase radio transmission power (within legal limits)
- Use a higher gain antenna for longer range
- Check for radio frequency interference
- Consider using a 4G/LTE link for longer range operations

## Advanced Configuration

### RTK Injection via MAVLink
For companion computer setups, RTCM corrections can be injected via MAVLink:
```python
# Example code for injecting RTCM via MAVLink
def inject_rtcm_via_mavlink(mavlink_connection, rtcm_data):
    for i in range(0, len(rtcm_data), 180):
        chunk = rtcm_data[i:i+180]
        mavlink_connection.mav.gps_rtcm_data_send(
            0,  # target system
            0,  # target component
            len(chunk),  # length of data
            0,  # sequence number
            bytearray(chunk)  # RTCM data
        )
```

### Dual-Antenna Systems for Heading
For enhanced heading accuracy without magnetometers:
- Install two GPS antennas with a separation of >30cm
- Configure the second antenna in u-center
- Set `GPS_YAW_OFFSET` parameter to match the physical installation

## References
- [PX4 RTK GPS Guide](https://docs.px4.io/master/en/gps_compass/rtk_gps.html)
- [u-blox ZED-F9P Integration Manual](https://www.u-blox.com/sites/default/files/ZED-F9P_IntegrationManual_%28UBX-18010802%29.pdf)
- [QGroundControl RTK Setup](https://docs.qgroundcontrol.com/master/en/SettingsView/RTK.html)