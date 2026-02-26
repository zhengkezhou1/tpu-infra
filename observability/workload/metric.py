import time
from libtpu.sdk import tpumonitoring
from prometheus_client import start_http_server, Gauge
from prometheus_client.core import REGISTRY, GaugeMetricFamily, HistogramMetricFamily
import traceback

CHIP_LABELS = ['chip']
CORE_TYPE_ID_LABELS = ['core_type_id']

DUTY_CYCLE = Gauge('libtpu_duty_cycle_pct', 'TPU Duty Cycle Percentage', CHIP_LABELS)
HBM_USAGE = Gauge('libtpu_hbm_usage_bytes', 'TPU HBM Usage in Bytes', CHIP_LABELS)
TENSORCORE_UTIL = Gauge('libtpu_tensorcore_utilization_ratio', 'TPU Tensor Core Utilization Ratio', CHIP_LABELS)
HBM_CAPACITY_TOTAL = Gauge('libtpu_hbm_capacity_total', 'TPU HBM Capacity Total in Bytes', CHIP_LABELS)
HLO_QUEUE_SIZE = Gauge('libtpu_hlo_queue_size', 'Number of instructions in the HLO queue', CORE_TYPE_ID_LABELS)
STAT_LABELS = ['core_type_id', 'stat']
HLO_EXECUTION_TIMING = Gauge('libtpu_hlo_execution_timing', 'HLO execution timing in microseconds', STAT_LABELS)

def main():
    start_http_server(9000)
    print("TPU Metrics Server started on port 9000")
    print("Monitoring all specified TPU metrics...")

    while True:
        try:
            metrics_with_chips = {
                "duty_cycle_pct": DUTY_CYCLE,
                "hbm_capacity_usage": HBM_USAGE,
                "tensorcore_util": TENSORCORE_UTIL,
                "hbm_capacity_total": HBM_CAPACITY_TOTAL,
                # "libtpu_hlo_queue_size": HLO_QUEUE_SIZE,
                # "libtpu_hlo_execution_timing": HLO_EXECUTION_TGIMING,
            }

            for api_name, prometheus_gauge in metrics_with_chips.items():
                data = tpumonitoring.get_metric(api_name).data()
                if isinstance(data, list):
                    for i, val in enumerate(data):
                        try:
                            prometheus_gauge.labels(chip=str(i)).set(float(val))
                        except (ValueError, TypeError):
                            print(f"Warning: Non-numeric data for {api_name} chip {i}: {val}. Skipping.")
                elif data is not None:
                    try:
                        prometheus_gauge.labels(chip="0").set(float(data))
                    except (ValueError, TypeError):
                         print(f"Warning: Non-numeric data for {api_name} single value: {data}. Skipping.")
                else:
                    print(f"Info: No data returned for {api_name}.")

            hlo_queue_data = tpumonitoring.get_metric("hlo_queue_size").data()
            if isinstance(hlo_queue_data, list):
                for item in hlo_queue_data:
                    if isinstance(item, str) and ':' in item:
                        core_id, value_str = item.split(':', 1)
                        try:
                            HLO_QUEUE_SIZE.labels(core_type_id=core_id.strip()).set(float(value_str.strip()))
                        except (ValueError, TypeError):
                            print(f"Warning: Could not parse HLO queue size data: '{item}'. Skipping.")
                    else:
                        print(f"Warning: Unexpected format for HLO queue size item: '{item}'. Skipping.")
            elif hlo_queue_data is not None:
                print(f"Warning: Unexpected format for HLO queue size data: {hlo_queue_data}. Expected a list.")
            else:
                print(f"Info: No data returned for hlo_queue_size.")

            hlo_execution_timing_data = tpumonitoring.get_metric("hlo_execution_timing").data()
            if isinstance(hlo_execution_timing_data, list):
                stat_names = ['mean', 'p50', 'p90', 'p95', 'p999']
                for item in hlo_execution_timing_data:
                    if isinstance(item, str) and ',' in item:
                        parts = [p.strip() for p in item.split(',')]
                        if len(parts) >= 6:
                            core_id = parts[0]
                            for stat_name, val_str in zip(stat_names, parts[1:6]):
                                try:
                                    HLO_EXECUTION_TIMING.labels(core_type_id=core_id, stat=stat_name).set(float(val_str))
                                except (ValueError, TypeError):
                                    print(f"Warning: Could not parse HLO execution timing value: core={core_id}, stat={stat_name}, val='{val_str}'. Skipping.")
                        else:
                            print(f"Warning: Unexpected number of fields for HLO execution timing: '{item}'. Skipping.")
                    else:
                        print(f"Warning: Unexpected format for HLO execution timing item: '{item}'. Skipping.")
            elif hlo_execution_timing_data is not None:
                print(f"Warning: Unexpected format for HLO execution timing data: {hlo_execution_timing_data}. Expected a list.")
            else:
                print(f"Info: No data returned for hlo_execution_timing.")

        except Exception as e:
            print(f"Error fetching or updating metrics: {e}")
            traceback.print_exc()

        time.sleep(1)

if __name__ == "__main__":
    main()
