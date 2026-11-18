# 开发与数据复现

普通读者只需打开在线地图。Python 和图像处理依赖主要用于修改或重新生成地图数据。

## 本地查看地图

在项目根目录启动 Python 自带的静态文件服务器：

```bash
python -m http.server 8000
```

然后在浏览器打开 `http://localhost:8000`。地图入口是 [`index.html`](../index.html)。查看矢量静态图可打开 [`vector_preview.svg`](../vector_preview.svg)。

## 重新生成数据

数据生成脚本需要 Python 和相应库。项目使用的处理库包括 OpenCV、NumPy、Pandas、Pillow、Shapely、SciPy 和 scikit-image。OCR 提取步骤还需要 Tesseract，并安装中文和英文语言包。不同脚本的实际依赖可能不同。

一个常见的数据处理顺序如下：

1. 从基础地图图像提取行政区、标记和文字候选。
2. 提取海域、邻国边界和跨境河流候选。
3. 使用官方高清分块图像生成精细海岸线。
4. 生成共享行政区边界和水系展示数据。
5. 构建坐标目录与统计数据。

对应命令如下。请在项目根目录运行，并按自己的输入文件位置调整路径：

```bash
python extract_map_features.py input_map.png --out .
python extract_context_layers.py
python tools/extract_official_coastline_tiled.py --source-dir "path/to/official_map"
python build_administrative_boundaries.py
python extract_hydrology_styles.py
python tools/build_coordinate_catalog.py
```

精细海岸线工具需要额外的官方高清分块图像目录；示例中的 `path/to/official_map` 只是占位路径，该目录没有随仓库提供。其他参数可查看相应脚本说明或运行 `--help`。

提取脚本会生成候选几何，边缘、符号和图像质量都可能造成误差。确认数据前应对照源图和审阅材料，不要把候选结果直接当作已核实事实。

## 使用海岸线审阅页面

从项目根目录运行：

```bash
python tools/coastline_pixel_audit.py
python -m http.server 8765
```

再打开 `http://127.0.0.1:8765/tools/coastline_review.html`。页面可以放大地图、查看原图像素并记录人工判定。完整说明见[海岸线审阅工具文档](../tools/README.md)。

## 相关源文件

- [`extract_map_features.py`](../extract_map_features.py)：提取地图要素和 OCR 候选。
- [`extract_context_layers.py`](../extract_context_layers.py)：提取海域、邻国边界和跨境河流候选。
- [`build_administrative_boundaries.py`](../build_administrative_boundaries.py)：生成共享边界和大区标注。
- [`extract_hydrology_styles.py`](../extract_hydrology_styles.py)：处理河流、季节性河流和运河展示层。
- [`tools/`](../tools/)：包含海岸线分块、坐标转换、地形与行政区统计等工具。
- [`data/`](../data/)：保存地图图层、名称目录、模型结果和坐标参数。
- [`FEATURE_EXTRACTION_REPORT.md`](../FEATURE_EXTRACTION_REPORT.md)：说明首轮提取输入、结果和质量限制。

## 坐标与数据安全

地图图层默认使用原图像素坐标。重新生成或转换数据前，先阅读[数据、坐标与精度边界](DATA_AND_LIMITS.md)，并确认脚本的输入、输出路径。项目坐标转换结果不是 WGS 84 / EPSG:4326。