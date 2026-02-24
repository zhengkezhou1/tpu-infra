import time
from libtpu.sdk import tpumonitoring
from prometheus_client import start_http_server, Gauge

# 定义 Prometheus 指标
DUTY_CYCLE = Gauge('tpu_duty_cycle_pct', 'TPU Duty Cycle Percentage', ['core'])
HBM_USAGE = Gauge('tpu_hbm_usage_bytes', 'TPU HBM Usage in Bytes', ['core'])

def main():
    # 启动 Prometheus HTTP 服务
    start_http_server(8000)
    print("🚀 TPU Metrics Server started on port 8000")
    print("Monitoring TPU duty cycle and HBM usage...")

    while True:
        try:
            # 获取 TPU 指标
            duty = tpumonitoring.get_metric("duty_cycle_pct").data()
            hbm = tpumonitoring.get_metric("hbm_capacity_usage").data()
            
            # 更新 Prometheus 指标
            for i, val in enumerate(duty):
                DUTY_CYCLE.labels(core=str(i)).set(val)
            for i, val in enumerate(hbm):
                HBM_USAGE.labels(core=str(i)).set(val)
            
            # 打印到日志（可选）
            # hbm_gb = [f"{float(v) / (1024**3):.2f}GB" for v in hbm]
            # print(f"Duty Cycle: {duty} | HBM: {hbm_gb}")
            
        except Exception as e:
            print(f"Error fetching metrics: {e}")
            
        time.sleep(2)  # 每 2 秒抓取一次

if __name__ == "__main__":
    main()
