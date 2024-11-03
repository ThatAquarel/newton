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

#define CE D0
#define CSN D7
const byte address[6] = "00001";
RF24 radio(CE, CSN);

#define ACCEL 0x53
#define GYRO 0x68

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

unsigned long prev_time;

void setup()
{
#ifdef DEBUG
  Serial.begin(115200);
#endif

  pixels.begin();
  pinMode(POWER, OUTPUT);
  digitalWrite(POWER, HIGH);

  radio_init();

  Wire.begin();
  accel_init();
  gyro_init();

  show_led(64, 64, 64);
  delay(100);

  prev_time = micros();
}

void radio_init()
{
  radio.begin();
  radio.setAutoAck(false);
  radio.setDataRate(RF24_250KBPS); //(RF24_250KBPS|RF24_1MBPS|RF24_2MBPS)
  radio.setPALevel(RF24_PA_MIN);   //(RF24_PA_MIN|RF24_PA_LOW|RF24_PA_HIGH|RF24_PA_MAX)
  radio.setPayloadSize(sizeof(payload));

  radio.openWritingPipe(address);
  radio.stopListening();
}

void accel_init()
{
  write_addr(ACCEL, 0x2D, 0b00001000); // POWER_CTL
  write_addr(ACCEL, 0x31, 0b00001011); // DATA_FORMAT, FULL_RES, +/- 16g
}

void accel_read()
{
  Wire.beginTransmission(ACCEL);
  Wire.write(0x32);
  Wire.endTransmission(false);
  Wire.requestFrom(ACCEL, 6, true);

  payload.accel_x = (Wire.read() | Wire.read() << 8);
  payload.accel_y = (Wire.read() | Wire.read() << 8);
  payload.accel_z = (Wire.read() | Wire.read() << 8);
}

void gyro_init()
{
  write_addr(GYRO, 0x16, 0b00011001); // DLPF, FS_SEL=3
  write_addr(GYRO, 0x15, 0b00000000); // SMPLRT_DIV 0, 1000Hz sampling rate
}

void gyro_read()
{
  Wire.beginTransmission(GYRO);
  Wire.write(0x1D);
  Wire.endTransmission(false);
  Wire.requestFrom(GYRO, 6, true);

  payload.gyro_x = (Wire.read() << 8 | Wire.read());
  payload.gyro_y = (Wire.read() << 8 | Wire.read());
  payload.gyro_z = (Wire.read() << 8 | Wire.read());
}

void write_addr(char addr, char register_addr, char data)
{
  Wire.beginTransmission(addr);
  Wire.write(register_addr);
  Wire.write(data);
  Wire.endTransmission();
}

void show_led(byte r, byte g, byte b)
{
  pixels.clear();
  pixels.setPixelColor(0, pixels.Color(r, g, b));
  pixels.show();
}

unsigned int sync;

void loop()
{
  payload.dt = micros() - prev_time;
  prev_time = micros();

  show_led(0, sync >> 4, 0);
  if (sync > 1024)
  {
    sync = 0;
  }
  sync++;

  payload.sync = sync;
  accel_read();
  gyro_read();

  byte res = radio.write(&payload, sizeof(payload));

  if (!res)
  {
    show_led(64, 0, 0);
  }

#ifdef DEBUG
  Serial.print(payload.dt);
  Serial.print("\t");
  Serial.print(payload.accel_x);
  Serial.print("\t");
  Serial.print(payload.accel_y);
  Serial.print("\t");
  Serial.println(payload.accel_z);
  Serial.print("\t");
  Serial.print(payload.gyro_x);
  Serial.print("\t");
  Serial.print(payload.gyro_y);
  Serial.print("\t");
  Serial.print(payload.gyro_z);
  Serial.println("\t");
#endif
}
