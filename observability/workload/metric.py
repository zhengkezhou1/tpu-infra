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
HLO_QUEUE_SIZE = Gauge('libtpu_hlo_queue_size', 'HLO Queue Size for CoreType-CoreID', CORE_TYPE_ID_LABELS)


class LibtpuHistogramCollector:
    """Custom collector that builds histogram snapshots from libtpu SDK data,
    avoiding the accumulation problem of prometheus_client Histogram.observe()."""

    HISTOGRAMS = [
        {
            'name': 'libtpu_buffer_transfer_latency',
            'doc': 'Network transfer latencies for megascale multislice traffic',
            'api': 'buffer_transfer_latency',
            'label_key': 'buffer_size',
            'buckets': [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, float('inf')],
        },
        {
            'name': 'libtpu_hlo_exec_timing',
            'doc': 'Granular performance insights into the HLO compiled binary execution status',
            'api': 'hlo_exec_timing',
            'label_key': 'core_type_id',
            'buckets': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, float('inf')],
        },
        {
            'name': 'libtpu_collective_e2e_latency',
            'doc': 'End-to-end collective latency over DCN in microseconds',
            'api': 'collective_e2e_latency',
            'label_key': 'transfer_size_op',
            'buckets': [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, float('inf')],
        },
        {
            'name': 'libtpu_grpc_tcp_min_round_trip_times',
            'doc': 'Distribution of the minimum Round Trip Times (RTT) on TCP connections by gRPC',
            'api': 'grpc_tcp_min_round_trip_times',
            'label_key': None,
            'buckets': [5, 10, 20, 50, 100, 200, 500, float('inf')],
        },
        {
            'name': 'libtpu_grpc_tcp_delivery_rates',
            'doc': 'Cumulative distribution of the recent throughput of TCP connections by gRPC',
            'api': 'grpc_tcp_delivery_rates',
            'label_key': None,
            'buckets': [1000, 5000, 10000, 20000, 50000, 100000, float('inf')],
        },
    ]

    def describe(self):
        """Return empty to indicate this is a non-self-describing collector."""
        return []

    def collect(self):
        for h in self.HISTOGRAMS:
            try:
                data_raw = tpumonitoring.get_metric(h['api']).data()
                if not isinstance(data_raw, list):
                    continue

                label_key = h['label_key']
                label_names = [label_key] if label_key else []
                hist = HistogramMetricFamily(h['name'], h['doc'], labels=label_names)
                buckets_upper = [b for b in h['buckets'] if b != float('inf')]

                # Group raw values by label
                groups = {}  # label_value -> [float values]
                for item_str in data_raw:
                    if not isinstance(item_str, str):
                        continue
                    parts = [p.strip().strip("'") for p in item_str.split(',')]

                    if label_key and len(parts) > 1:
                        label_val = parts[0]
                        raw_vals = parts[1:]
                    else:
                        label_val = None
                        raw_vals = parts

                    values = []
                    for v in raw_vals:
                        try:
                            values.append(float(v))
                        except (ValueError, TypeError):
                            pass

                    groups.setdefault(label_val, []).extend(values)

                for label_val, values in groups.items():
                    # Build cumulative bucket counts
                    bucket_counts = []
                    for upper in buckets_upper:
                        bucket_counts.append((str(upper), sum(1 for v in values if v <= upper)))
                    bucket_counts.append(('+Inf', len(values)))

                    sum_value = sum(values)
                    labels = [label_val] if label_key else []
                    hist.add_metric(labels, buckets=bucket_counts, sum_value=sum_value)

                yield hist
            except Exception as e:
                print(f"Warning: Failed to collect {h['name']}: {e}")


REGISTRY.register(LibtpuHistogramCollector())


def main():
    start_http_server(8000)
    print("TPU Metrics Server started on port 8000")
    print("Monitoring all specified TPU metrics...")

    while True:
        try:
            metrics_with_chips = {
                "duty_cycle_pct": DUTY_CYCLE,
                "hbm_capacity_usage": HBM_USAGE,
                "tensorcore_util": TENSORCORE_UTIL,
                "hbm_capacity_total": HBM_CAPACITY_TOTAL,
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

            # Histograms are collected by LibtpuHistogramCollector on each scrape

        except Exception as e:
            print(f"Error fetching or updating metrics: {e}")
            traceback.print_exc()

        time.sleep(1)

if __name__ == "__main__":
    main()
