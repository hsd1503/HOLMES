
from resnet1d.resnet1d import ResNet1D
import ensemble_profiler as profiler
from pathlib import Path

# ECG
n_channel = 1
base_filters = 64
kernel_size = 16
n_classes = 2
n_block = 2
model = ResNet1D(in_channels=n_channel,
                 base_filters=base_filters,
                 kernel_size=kernel_size,
                 stride=2,
                 n_block=n_block,
                 groups=base_filters,
                 n_classes=n_classes,
                 downsample_gap=max(n_block//8, 1),
                 increasefilter_gap=max(n_block//4, 1),
                 verbose=False)

time, num_queries = profiler.calculate_throughput([model])
throughput = float(num_queries)/time
print("Throughput is : {} QPS".format(throughput))
