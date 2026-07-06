# Local Simulation

目标：不用真集群也能先练推理服务的工程结构。

## 可练内容

- Docker Compose 多 worker；
- router；
- health check；
- rolling restart；
- Prometheus/Grafana；
- 压测；
- worker crash recovery。

## 第一版目标

用多个轻量 worker 模拟推理服务，先把 API、路由、压测和指标链路打通。
