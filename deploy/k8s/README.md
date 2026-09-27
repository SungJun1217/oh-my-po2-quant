# Kubernetes experiment environment

Access: this Mac → `jun@100.76.167.82` (jump) → `master-sjlee@192.168.3.121` (k8s master, kubectl).
GPU workers: worker1/3/4 each have **one physical RTX 3090** (24 GB, driver 570, CUDA 12.8, cc 8.6 → `f32_chunked` conv backend),
advertised as `nvidia.com/gpu: 4` through device-plugin **time-slicing** (`replicas: 4`, max 1 per pod). Slices share memory and compute
with no isolation, so check `nvidia-smi` memory before a run. worker2 (also 1× GPU) is cordoned.
Shared cluster: create resources only in namespace `po2-quant`.

```bash
# on the master
kubectl apply -f deploy/k8s/storage.yaml     # NFS PV/PVC on /mnt/sdb2/workspaces/po2-quant (dir must exist)
kubectl apply -f deploy/k8s/dev-pod.yaml     # apply each file separately
kubectl -n po2-quant exec -it po2-dev -- bash
```

Layout on the volume: `/data/datasets` (`$PO2_DATA_ROOT`), `/data/runs` (`$PO2_RUNS_ROOT`), `/data/cache`, `/data/venv`, `/data/work`.
ImageNet must be downloaded by the user (license) into `/data/datasets/imagenet`.
