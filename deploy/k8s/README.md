# Kubernetes experiment environment

Access: this Mac → `jun@100.76.167.82` (jump) → `master-sjlee@192.168.3.121` (k8s master, kubectl).
GPU workers: worker1/3/4 (4× RTX 3090 each, driver 570, CUDA 12.8; cc 8.6 → `f32_chunked` conv backend).
Shared cluster: create resources only in namespace `po2-quant`.

```bash
# on the master
kubectl apply -f deploy/k8s/storage.yaml     # NFS PV/PVC on /mnt/sdb2/workspaces/po2-quant (dir must exist)
kubectl apply -f deploy/k8s/dev-pod.yaml
kubectl -n po2-quant exec -it po2-dev -- bash
```

Layout on the volume: `/data/datasets` (`$PO2_DATA_ROOT`), `/data/runs` (`$PO2_RUNS_ROOT`), `/data/cache`, `/data/venv`, `/data/work`.
ImageNet must be downloaded by the user (license) into `/data/datasets/imagenet`.
