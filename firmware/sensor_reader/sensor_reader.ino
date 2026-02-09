/*
 * TB Breath Analyzer - Sensor Reader
 * ESP32 DevKit V1 Firmware
 * 
 * Reads MQ-3, MQ-135, MQ-2 sensors at 100ms intervals.
 * Outputs CSV format via Serial.
 */

struct SensorConfig {
    const char* name;
    uint8_t pin;
};

const SensorConfig SENSORS[] = {
    {"MQ3",   34},
    {"MQ135", 35},
    {"MQ2",   32}
};

const uint8_t NUM_SENSORS = sizeof(SENSORS) / sizeof(SENSORS[0]);
const unsigned long SAMPLE_INTERVAL_MS = 100;

unsigned long lastSampleTime = 0;

void setup() {
    Serial.begin(115200);
    while (!Serial) { delay(10); }
    
    for (uint8_t i = 0; i < NUM_SENSORS; i++) {
        pinMode(SENSORS[i].pin, INPUT);
    }
    
    Serial.println("TB Breath Analyzer Ready");
    Serial.print("Sensors: ");
    Serial.println(NUM_SENSORS);
}

void loop() {
    unsigned long currentTime = millis();
    
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
        lastSampleTime = currentTime;
        outputSensorData();
    }
}

void outputSensorData() {
    for (uint8_t i = 0; i < NUM_SENSORS; i++) {
        int value = analogRead(SENSORS[i].pin);
        Serial.print(value);
        if (i < NUM_SENSORS - 1) {
            Serial.print(",");
        }
    }
    Serial.println();
}
