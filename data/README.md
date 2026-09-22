# Data directory

此目录只保留说明文件。使用仓库内相对路径配置时，实际音频和数据库存放在：

```text
./data/
```

推荐子目录：

```text
raw/          原始音频，只读
processed/    切分或降噪后的音频
annotations/  人工标注
manifests/    JSONL 数据清单
state/        SQLite 等运行状态
```
