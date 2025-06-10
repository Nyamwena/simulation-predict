import json

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from datetime import datetime, timedelta
from flask import send_file
from io import BytesIO


def sigmoid(x, k=10, x0=0.5):
    return 1 / (1 + np.exp(-k * (x - x0)))


# Sensor definitions for simulation
SENSORS = {
    'mill_motor_air_temp': {'mean': 75, 'std': 5},
    'coal_feed_flow': {'mean': 3.75, 'std': 0.1},
    'mill_inlet_temp': {'mean': 185, 'std': 3},
    'mill_inlet_pressure': {'mean': 8500, 'std': 200},
    'mill_diff_pressure': {'mean': 4500, 'std': 300},
    'lubrication_oil_pressure': {'mean': 110, 'std': 5},
    'hydraulic_loading_pressure': {'mean': 4.5, 'std': 0.5},
    'sealing_air_pressure': {'mean': 12, 'std': 1},
    'motor_current': {'mean': 42, 'std': 4},
    'vibrations': {'mean': 2.0, 'std': 1.0},
    'mill_outlet_temp': {'mean': 95, 'std': 5},
    'machine_loading': {'mean': 90, 'std': 10},
}

# Global state storage
current_data = None
latest_export_path = None
app = Flask(__name__)

# Load trained LSTM model for inference
tf_model = tf.keras.models.load_model('best_lstm_model.keras')
scalers = joblib.load('scalers.pkl')

# You may need custom objects if any, or pass compile=False
# tf_model = tf.keras.models.load_model('best_lstm_model.keras', compile=False)

# app2/reporting.py
import pandas as pd
from datetime import datetime, timedelta

# app2/reporting.py
import pandas as pd
from datetime import datetime, timedelta

# Sample thresholds for sensors (you can adjust)
THRESHOLDS = {
    'mill_motor_air_temp': (55.0, 99.0),
    'coal_feed_flow': (3.5, 4.0),
    'mill_inlet_temp': (180.0, None),      # None means no upper bound
    'mill_inlet_pressure': (8000.0, None),
    'mill_diff_pressure': (None, 6000.0),  # None means no lower bound
    'mill_lubrication_oil_pressure': (None, 100.0),
    'mill_hydraulic_loading_pressure': (3.5, 5.5),
    'sealing_air_pressure': (10.0, None),
    'mill_motor_current': (36.0, 50.0),
    'machine_loading': (60.0, 120.0),
    'vibrations_velocity': (None, 4.5),
    'mill_outlet_temp': (None, 99.0)
}

RECOMMENDATIONS = {
    'mill_motor_air_temp': "Check cooling system and motor windings.",
    'coal_feed_flow': "Inspect coal feeders and pipes.",
    'mill_inlet_temp': "Check primary air temperature controls.",
    'mill_inlet_pressure': "Inspect PA fans and inlet ducting.",
    'mill_diff_pressure': "Check grinding zone for clogging or wear.",
    'mill_lubrication_oil_pressure': "Inspect gearbox lubrication system.",
    'mill_hydraulic_loading_pressure': "Check hydraulic rams and pumps.",
    'sealing_air_pressure': "Inspect seals and air piping.",
    'mill_motor_current': "Monitor motor loading and current draw.",
    'machine_loading': "Adjust generator/turbine demand.",
    'vibrations_velocity': "Inspect structural and gearbox mounts.",
    'mill_outlet_temp': "Monitor outlet mixture temperature."
}

ISSUES = {
    'mill_motor_air_temp': 'Overheating detected in motor.',
    'coal_feed_flow': 'Abnormal coal feed rate.',
    'mill_inlet_temp': 'Inlet temperature out of range.',
    'mill_inlet_pressure': 'Inlet pressure anomaly.',
    'mill_diff_pressure': 'Differential pressure anomaly.',
    'mill_lubrication_oil_pressure': 'Oil pressure low.',
    'mill_hydraulic_loading_pressure': 'Hydraulic pressure out of range.',
    'sealing_air_pressure': 'Seal air pressure low.',
    'mill_motor_current': 'Motor current anomaly.',
    'machine_loading': 'Machine load anomaly.',
    'vibrations_velocity': 'Vibration levels high.',
    'mill_outlet_temp': 'Outlet temperature anomaly.'
}


def estimate_rul_linear(sensor_series, threshold):
    recent = sensor_series[-10:]
    x = np.arange(len(recent))
    y = recent.values
    if len(x) > 1 and np.ptp(y) != 0:
        a, b = np.polyfit(x, y, 1)
        if a != 0:
            steps = (threshold - y[-1]) / a
            return max(0, steps)
    return 0


def get_threshold_for_sensor(sensor):
    return THRESHOLDS.get(sensor)


def get_issue_for_sensor(sensor):
    return ISSUES.get(sensor, 'Anomaly detected')


def get_recommendation_for_sensor(sensor):
    return RECOMMENDATIONS.get(sensor, 'Check system')


def generate_equipment_report(df):
    report = {}

    # Ensure 'timestamp' is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Time step in hours between rows
    ts = df['timestamp']
    delta = 1.0
    if len(ts) >= 2:
        delta = (ts.iloc[1] - ts.iloc[0]).total_seconds() / 3600.0

    current_time = ts.iloc[-1]
    core_sensors = list(THRESHOLDS.keys())

    for sensor in core_sensors:
        if sensor not in df.columns:
            continue

        series = df.set_index('timestamp')[sensor]
        threshold = get_threshold_for_sensor(sensor)
        if threshold is None:
            continue

        # Identify all times sensor exceeds threshold
        # Unpack lower/upper bounds
        lower, upper = threshold
        failures = pd.Series(False, index=series.index)
        if lower is not None:
            failures |= (series < lower)
        if upper is not None:
            failures |= (series > upper)
        failure_times = series[failures].index
        if len(failure_times) == 0:
            # No anomalies for this sensor; skip reporting
            continue

                # Forecast linear trend to the next threshold breach
        current_val = series.iloc[-1]
        # Determine relevant bound for prediction
        lower, upper = threshold
        if lower is not None and current_val < lower:
            target = lower
        elif upper is not None and current_val > upper:
            target = upper
        else:
            continue
        # Estimate steps to reach the bound
        steps = estimate_rul_linear(series, target)
        estimated_rul = steps * delta
        T_failure = current_time + pd.Timedelta(hours=estimated_rul)
        tf_str = T_failure.isoformat()

        issue = get_issue_for_sensor(sensor)
        rec = get_recommendation_for_sensor(sensor)
        current_val = round(series.iloc[-1], 2)

        report[sensor] = {
            'equipment': sensor.replace('_', ' ').title(),
            'issue': issue,
            'recommendation': rec,
            'current_value': current_val,
            'threshold': threshold,
            'estimated_rul': estimated_rul,
            'T_failure': tf_str
        }

    return report, df


def export_report_to_excel(report_dict, df_with_rul, filename="equipment_status_report.xlsx"):
    report_df = pd.DataFrame.from_dict(report_dict, orient='index')
    with pd.ExcelWriter(filename) as writer:
        report_df.to_excel(writer, sheet_name='Report')
        df_with_rul.to_excel(writer, sheet_name='RUL_Data', index=False)
    return filename

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/real-time-data')
def real_time_data():
    global current_data
    now = datetime.utcnow()

    # If no simulation run yet, use the means
    if current_data is None:
        sample = {k: v['mean'] for k, v in SENSORS.items()}
    else:
        # Pull only the raw sensor columns from the last row
        sample = {
            k: float(current_data.iloc[-1][k])
            for k in SENSORS.keys()
        }

    status = 'Normal'
    return jsonify(
        success=True,
        status=status,
        timestamp=now.isoformat(),
        **sample
    )


@app.route('/simulate', methods=['POST'])
def simulate():
    global current_data
    global latest_report, latest_df_with_rul
    payload = request.json
    duration = int(payload.get('duration', 1))
    periods = duration * 24
    t = np.linspace(0, 1, periods)
    base_time = datetime.utcnow() - timedelta(hours=periods)

    # 1) GENERATE RAW SENSOR READINGS
    anomaly_rate = 0.05
    anomaly_idxs = set(np.random.choice(periods, size=int(periods * anomaly_rate), replace=False))
    records = []
    for i in range(periods):
        ts = base_time + timedelta(hours=i)
        drift = sigmoid(t[i], k=12, x0=0.5)
        row = {'timestamp': ts}
        for name, p in SENSORS.items():
            base, std = p['mean'], p['std']
            daily = np.sin(2 * np.pi * (i % 24) / 24) * std * 0.5
            val = base + (drift - 0.5) * std + daily + np.random.normal(scale=std)
            if i in anomaly_idxs:
                val += std * (1 + np.random.rand() * 2)
            row[name] = float(val)
        records.append(row)

    df = pd.DataFrame(records)
    current_data = df

    # 2) FEATURE ENGINEERING (match training pipeline)
    sensors = list(SENSORS.keys())
    for s in sensors:
        df[f"{s}_diff"] = df[s].diff().fillna(0)
        df[f"{s}_roll_mean"] = df[s].rolling(window=24, min_periods=1).mean()

    # 3) SCALE FEATURES USING SAVED SCALERS
    feature_cols = sensors + [f"{s}_diff" for s in sensors] + [f"{s}_roll_mean" for s in sensors]
    X_raw = df[feature_cols].values
    X_scaled = np.zeros_like(X_raw)
    for idx, col in enumerate(feature_cols):
        X_scaled[:, idx] = scalers[col].transform(X_raw[:, [idx]]).flatten()

    # 4) SEQUENCE CREATION
    seq_len = 24
    X_seq = np.stack([X_scaled[i:i + seq_len] for i in range(len(df) - seq_len + 1)])

    # 5) PREDICTION
    preds = tf_model.predict(X_seq)
    pad = np.zeros((seq_len - 1, preds.shape[1]))
    all_preds = np.vstack([pad, preds])

    # 6) BUILD RESPONSE
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    timestamps = df['timestamp'].apply(lambda x: x.isoformat()).tolist()
    sensor_data = {name: df[name].tolist() for name in sensors}
    statuses = ['Anomaly' if row.max() > 0.5 else 'Normal' for row in all_preds]

    predictions = []
    for ts, row, status in zip(timestamps, all_preds, statuses):
        m = float(row.max())
        predictions.append({
            'timestamp': ts,
            'status': status,
            'confidence': m,
            'probabilities': {'normal': 1 - m, 'anomaly': m}
        })

    summary = {
        'total_periods': periods,
        'normal_count': statuses.count('Normal'),
        'anomaly_count': statuses.count('Anomaly')
    }
    report, df_with_rul = generate_equipment_report(df)
    latest_report, latest_df_with_rul = generate_equipment_report(df)
    print("==== Equipment Report ====")
    print(json.dumps(report, indent=2))
    return jsonify(
        success=True,
        sensor_data={'timestamps': timestamps, **sensor_data},
        summary=summary,
        recommendations=[],
        predictions=predictions,
        equipment_report=report
    )


@app.route('/download-report', methods=['POST'])
def download_report():
    global latest_report, latest_df_with_rul
    if latest_report and latest_df_with_rul is not None:
        filename = "equipment_status_report.xlsx"
        export_report_to_excel(latest_report, latest_df_with_rul, filename)
        return send_file(filename, as_attachment=True)
    else:
        return "No report available to download.", 400
@app.route('/export-report', methods=['POST'])
def export_report():
    data = request.json
    # simple text report
    report = [f"{k}: {v}" for k, v in data.items()]
    return jsonify(success=True, report_content="\n".join(report))




if __name__ == '__main__':
    app.run(debug=True, port=5050)
