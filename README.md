# 潼国地图特征提取包

> **项目状态：**目前处于项目协作与内测阶段，内容和功能仍在持续完善。


这是对用户提供的 2048×1448 PNG 地图进行的第一轮半自动矢量化结果。

## 快速查看
直接用浏览器打开 `index.html`：
- 鼠标滚轮缩放
- 鼠标拖拽平移
- 点击行政区查看代码
- 右侧可开关河流、湖泊、点标记和文字图层

`vector_preview.svg` 是独立矢量预览，可用浏览器、Inkscape 或 Illustrator 打开。

## 数据文件
- `data/administrative_regions.geojson`：行政区候选面
- `data/national_outline.geojson`：国土外轮廓候选
- `data/hydrology_lines.geojson`：河流候选折线
- `data/lakes.geojson`：湖泊候选面
- `data/settlement_markers.geojson`：黑色/红色点标记候选
- `data/region_code_labels.geojson`：行政区代码标注点
- `data/region_codes.csv`：代码与组件关联表
- `data/ocr_candidates.csv`：OCR 文字候选及置信度、像素外接框
- `data/map_metadata.json`：地图标题、版本和坐标说明

## 可复现脚本
`extract_map_features.py` 可从输入图片重新生成行政区面、国土轮廓、水系线、湖泊、点标记、代码标注、OCR 候选和诊断掩膜。需要 Python、OpenCV、NumPy、Pandas、Pillow、Shapely、scikit-image、pytesseract 和 Tesseract 中文/英文语言包。

运行示例：
```bash
python extract_map_features.py input_map.png --out map_extraction
```

## 重要说明
- 预览由矢量路径、点和文字独立绘制，不使用原始位图作为显示底图。
- 坐标是原图像素坐标，不是经纬度。
- 自动提取结果是候选数据，不是已完成测绘或法律边界数据。
- 行政区代码已做视觉辅助转录，仍应逐项核验。



