# 02 Quantization Comparison

目标：比较不同量化格式在 4060 上的速度、显存和质量。

## 对比对象

- GGUF Q4 / Q5 / Q8；
- AWQ；
- GPTQ；
- EXL2。

## 固定变量

- 同一模型；
- 同一 prompt；
- 同一 max tokens；
- 同一 context length；
- 同一后端版本。

## 输出

- 速度对比；
- 显存对比；
- 简单质量对比；
- 推荐量化格式。
