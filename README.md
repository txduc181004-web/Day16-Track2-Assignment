# Lab 16 – Cloud AI Environment Setup (AWS)

Trần Xuân Đức · Luồng CPU + LightGBM · AWS `us-east-1` · 02/10/2026. Hướng dẫn gốc của lab: [README_aws.md](README_aws.md).

## Báo cáo

1. Hạ tầng gồm 27 tài nguyên được dựng bằng Terraform: VPC, Bastion `t3.micro`, NAT Gateway, ALB và Compute Node trong private subnet. Lần apply đầu mất khoảng 3 phút.
2. Compute Node dùng `c7i-flex.large` (2 vCPU, 4 GB RAM) thay cho `t3.medium`. Lý do là tài khoản Free plan không cho chạy instance ngoài Free Tier. Thay đổi này nằm trong `terraform/terraform.tfvars`.
3. Training mất 2.53 s cho khoảng 200,000 dòng. LightGBM dùng hết 2 vCPU (`python3` 200% CPU) nhưng chỉ dùng khoảng 450 MB RAM, nên CPU là đủ và không cần GPU.
4. AUC-ROC đạt 0.9753. Accuracy 99.95% ít ý nghĩa, vì đoán mọi giao dịch đều hợp lệ đã đạt 99.83%.
5. Ở ngưỡng 0.5, mô hình bắt được 76/98 giao dịch gian lận (Recall 0.78), và 95% cảnh báo là đúng (Precision 0.95).
6. Với tham số mặc định, validation loss tăng ngay sau cây đầu tiên nên early stopping dừng ở iteration 1 (AUC 0.93). Thêm `min_child_weight=1` và `reg_lambda=1` giúp mô hình ổn định, dừng ở iteration 49.
7. Inference 1 dòng mất 0.6 ms, đủ cho ứng dụng thời gian thực. Chạy theo lô thì đạt khoảng 519,000 dòng/s, rẻ hơn khoảng 300 lần cho mỗi dòng.
8. Billing chưa có dữ liệu (USD 0.00) do AWS cập nhật trễ, và chi phí được trừ vào credit $100 của Free plan (ước tính khoảng $0.10/giờ). `terraform destroy` đã xóa đủ 27 tài nguyên.

## Kết quả benchmark

| Metric | Kết quả |
| --- | --- |
| Thời gian load data | 1.028 s |
| Thời gian training | 2.526 s |
| Best iteration | 49 |
| AUC-ROC | 0.9753 |
| Accuracy | 0.9995 |
| F1-Score | 0.8539 |
| Precision | 0.9500 |
| Recall | 0.7755 |
| Inference latency (1 row) | 0.598 ms |
| Inference throughput (1000 rows) | 1.926 ms (~519,243 rows/s) |

Dataset: Credit Card Fraud Detection, gồm 284,807 dòng và 30 đặc trưng, tỷ lệ gian lận 0.17%. Dữ liệu được chia stratified thành train 70%, validation 10% và test 20%.

## Deliverables

| # | Nội dung | File |
| --- | --- | --- |
| 1 | Output `python3 benchmark.py` | [screenshots/01_benchmark_output.png](screenshots/01_benchmark_output.png) |
| 2 | Metrics đầy đủ | [benchmark_result.json](benchmark_result.json) (script: [benchmark.py](benchmark.py)) |
| 3 | CPU / RAM / Network | [02_top_cpu_usage.png](screenshots/02_top_cpu_usage.png), [03_free_ram_network.png](screenshots/03_free_ram_network.png) |
| 4 | Billing và tài nguyên phát sinh chi phí | [04_billing_bills.png](screenshots/04_billing_bills.png), [05_billing_credits.png](screenshots/05_billing_credits.png), [06_ec2_instances_running.png](screenshots/06_ec2_instances_running.png), [07_nat_gateway_available.png](screenshots/07_nat_gateway_available.png) |
| 5 | Mã nguồn Terraform đã chạy | [terraform.zip](terraform.zip) |
| 6 | Báo cáo ngắn | Mục **Báo cáo** ở trên |
