#include <SPI.h>
#include <Wire.h>

#include <RF24.h>
#include <nRF24L01.h>

#include <Adafruit_NeoPixel.h>

// #define DEBUG

#define POWER 11
#define PIN 12
#define NUMPIXELS 1
Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);

#define INTERVAL_MS_SIGNAL_LOST 50
#define INTERVAL_MS_SIGNAL_RETRY 250

#define CE D0
#define CSN D7
const byte address[6] = "00001";
RF24 radio(CE, CSN);

struct imu_payload
{
  unsigned int sync;

  unsigned long dt;

  uint16_t accel_x;
  uint16_t accel_y;
  uint16_t accel_z;

  uint16_t gyro_x;
  uint16_t gyro_y;
  uint16_t gyro_z;
};

imu_payload payload;

unsigned long last_signal_millis = 0;

void setup()
{
  Serial.begin(115200);

  pixels.begin();
  pinMode(POWER, OUTPUT);
  digitalWrite(POWER, HIGH);
  pixels.clear();

  radio_init();

  pixels.setPixelColor(0, pixels.Color(64, 64, 64));
  delay(100);
}

void radio_init()
{
  radio.begin();
  radio.setAutoAck(false);
  radio.setDataRate(RF24_250KBPS); //(RF24_250KBPS|RF24_1MBPS|RF24_2MBPS)
  radio.setPALevel(RF24_PA_MIN);   //(RF24_PA_MIN|RF24_PA_LOW|RF24_PA_HIGH|RF24_PA_MAX)
  radio.setPayloadSize(sizeof(payload));

  radio.openReadingPipe(0, address);
  radio.startListening();
}

void loop()
{
  unsigned long current_millis = millis();
  if (radio.available() > 0)
  {
    radio.read(&payload, sizeof(payload));

    Serial.write((uint8_t)0x00);
    Serial.write((uint8_t)0x7E);
    Serial.write((uint8_t *)&payload.dt, sizeof(unsigned long));
    Serial.write((uint8_t *)&payload.accel_x, sizeof(uint16_t));
    Serial.write((uint8_t *)&payload.accel_y, sizeof(uint16_t));
    Serial.write((uint8_t *)&payload.accel_z, sizeof(uint16_t));
    Serial.write((uint8_t *)&payload.gyro_x, sizeof(uint16_t));
    Serial.write((uint8_t *)&payload.gyro_y, sizeof(uint16_t));
    Serial.write((uint8_t *)&payload.gyro_z, sizeof(uint16_t));
    Serial.write((uint8_t)0x7D);
    Serial.write((uint8_t)0x00);

    last_signal_millis = current_millis;
    show_led(0, payload.sync >> 4, 0);
  }
  if (current_millis - last_signal_millis > INTERVAL_MS_SIGNAL_LOST)
  {
    show_led(64, 0, 0);
    delay(INTERVAL_MS_SIGNAL_RETRY);
  }
}

void show_led(byte r, byte g, byte b)
{
  pixels.clear();
  pixels.setPixelColor(0, pixels.Color(r, g, b));
  pixels.show();
}
