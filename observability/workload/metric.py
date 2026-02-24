import time
from libtpu.sdk import tpumonitoring
from prometheus_client import start_http_server, Gauge, Histogram
import traceback

CORE_LABELS = ['core']
CORE_TYPE_ID_LABELS = ['core_type_id']
BUFFER_SIZE_LABELS = ['buffer_size']
TRANSFER_SIZE_OP_LABELS = ['transfer_size_op']

DUTY_CYCLE = Gauge('libtpu_duty_cycle_pct', 'TPU Duty Cycle Percentage', CORE_LABELS)
HBM_USAGE = Gauge('libtpu_hbm_usage_bytes', 'TPU HBM Usage in Bytes', CORE_LABELS)

TENSORCORE_UTIL = Gauge('libtpu_tensorcore_util', 'TPU Tensor Core Utilization Percentage', CORE_LABELS)
HBM_CAPACITY_TOTAL = Gauge('libtpu_hbm_capacity_total', 'TPU HBM Capacity Total in Bytes', CORE_LABELS)
HLO_QUEUE_SIZE = Gauge('libtpu_hlo_queue_size', 'HLO Queue Size for CoreType-CoreID', CORE_TYPE_ID_LABELS)

BUFFER_TRANSFER_LATENCY = Histogram(
    'libtpu_buffer_transfer_latency',
    'Network transfer latencies for megascale multislice traffic',
    BUFFER_SIZE_LABELS,
    buckets=[100, 500, 1000, 2000, 5000, 10000, 20000, 50000, float('inf')]
)
HLO_EXEC_TIMING = Histogram(
    'libtpu_hlo_exec_timing',
    'Granular performance insights into the HLO compiled binary execution status',
    CORE_TYPE_ID_LABELS,
    buckets=[0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, float('inf')] # Assuming time in ms
)
COLLECTIVE_E2E_LATENCY = Histogram(
    'libtpu_collective_e2e_latency',
    'End-to-end collective latency over DCN in microseconds',
    TRANSFER_SIZE_OP_LABELS,
    buckets=[500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, float('inf')]
)
GRPC_TCP_MIN_RTT = Histogram(
    'libtpu_grpc_tcp_min_round_trip_times',
    'Distribution of the minimum Round Trip Times (RTT) on TCP connections by gRPC',
    [],
    buckets=[5, 10, 20, 50, 100, 200, 500, float('inf')] # Microseconds
)
GRPC_TCP_DELIVERY_RATES = Histogram(
    'libtpu_grpc_tcp_delivery_rates',
    'Cumulative distribution of the recent throughput of TCP connections by gRPC',
    [],
    buckets=[1000, 5000, 10000, 20000, 50000, 100000, float('inf')]
)

def parse_and_observe_histogram(histogram_metric, data_raw, label_key=None):
    """
    Parses raw data from tpumonitoring.get_metric().data() and observes values for a histogram.
    Expected data_raw format (list of strings):
    For labeled: ["'label_val', 'val1', 'val2', ...", ...]
    For unlabeled: ["'val1', 'val2', ...", ...] or ["val1", "val2", ...]
    """
    if not isinstance(data_raw, list):
        print(f"Warning: Expected list for histogram data, got {type(data_raw)}. Skipping.")
        return

    for item_str in data_raw:
        if not isinstance(item_str, str):
            print(f"Warning: Expected string item in histogram data, got {type(item_str)}. Skipping.")
            continue

        parts = [p.strip().strip("'") for p in item_str.split(',')]
        
        current_labels = {}
        values_to_observe = []

        if label_key and len(parts) > 1:
            current_labels[label_key] = parts[0]
            values_to_observe = parts[1:]
        else:
            values_to_observe = parts
        
        for val_str in values_to_observe:
            try:
                val = float(val_str)
                if current_labels:
                    histogram_metric.labels(**current_labels).observe(val)
                else:
                    histogram_metric.observe(val)
            except (ValueError, TypeError):
                print(f"Warning: Could not parse histogram value '{val_str}' for {histogram_metric._name}. Skipping.")


def main():
    start_http_server(8000)
    print("🚀 TPU Metrics Server started on port 8000")
    print("Monitoring all specified TPU metrics...")

    while True:
        try:
            metrics_with_cores = {
                "duty_cycle_pct": DUTY_CYCLE,
                "hbm_capacity_usage": HBM_USAGE,
                "tensorcore_util": TENSORCORE_UTIL,
                "hbm_capacity_total": HBM_CAPACITY_TOTAL,
            }

            for api_name, prometheus_gauge in metrics_with_cores.items():
                data = tpumonitoring.get_metric(api_name).data()
                if isinstance(data, list):
                    for i, val in enumerate(data):
                        try:
                            prometheus_gauge.labels(core=str(i)).set(float(val))
                        except (ValueError, TypeError):
                            print(f"Warning: Non-numeric data for {api_name} core {i}: {val}. Skipping.")
                elif data is not None:
                    # 处理tpumonitoring返回单个值的情况，假定属于core 0
                    try:
                        prometheus_gauge.labels(core="0").set(float(data))
                    except (ValueError, TypeError):
                         print(f"Warning: Non-numeric data for {api_name} single value: {data}. Skipping.")
                else:
                    print(f"Info: No data returned for {api_name}.")


            # HLO Queue Size - 假设tpumonitoring返回格式如 ["core-id: value", "core-id: value"]
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


            parse_and_observe_histogram(BUFFER_TRANSFER_LATENCY, tpumonitoring.get_metric("buffer_transfer_latency").data(), label_key='buffer_size')
            parse_and_observe_histogram(HLO_EXEC_TIMING, tpumonitoring.get_metric("hlo_exec_timing").data(), label_key='core_type_id')
            parse_and_observe_histogram(COLLECTIVE_E2E_LATENCY, tpumonitoring.get_metric("collective_e2e_latency").data(), label_key='transfer_size_op')
            parse_and_observe_histogram(GRPC_TCP_MIN_RTT, tpumonitoring.get_metric("grpc_tcp_min_round_trip_times").data())
            parse_and_observe_histogram(GRPC_TCP_DELIVERY_RATES, tpumonitoring.get_metric("grpc_tcp_delivery_rates").data())

        except Exception as e:
            print(f"Error fetching or updating metrics: {e}")
            traceback.print_exc()

        time.sleep(1)

if __name__ == "__main__":
    main()
