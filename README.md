# 湩国地图特征提取包

> **项目状态：**目前处于项目协作与内测阶段，内容和功能仍在持续完善。


这是对用户提供的 2048×1448 PNG 地图进行的第一轮半自动矢量化结果。

## 快速查看
直接用浏览器打开 `index.html`：
- 鼠标滚轮缩放
- 鼠标拖拽平移
- 点击行政区查看代码，并以缓动动画放大到该区域
- 右侧按“边界、水系、城市标记、文字标注”分组；父项默认收起，父项开关可控制整组子图层
- 预览会平滑显示河流线条与湖泊边缘
- 南部两段尺刻符号单独作为运河图层；东南沿海网状水道仍按河口河网显示
- 邻国蓝色虚线河段按底图重新描绘并单独显示为季节性河流；原图国境内未发现同类虚线
- 主页面使用 Canvas；缩小时简化线条并隐藏点和文字，放大时只绘制当前视口附近的图形
- 地图可进一步放大；海陆填充边缘使用多档几何精度，放大进度约 65% 时达到最高精度；宏观视图下以海水色覆盖过小的海岛
- 海岸由陆地与海水的颜色分界呈现，不额外绘制灰色描边；海岛图层可单独开关
- 省界由相邻行政区共同生成一条中线，使用细虚线；界河处沿河道中线，界湖处采用直线
- 湩国外缘和邻国边界使用较粗实线
- 画布右侧扩展为更宽的海域，减少海侧留白

`vector_preview.svg` 是独立矢量预览，可用浏览器、Inkscape 或 Illustrator 打开。

## 数据文件
- `Administrative-Division-Map-of-Donh-2026.png`: 高清原图（8192 ? 5792，用于运河中心线和刻度描绘）
- `data/administrative_regions.geojson`：行政区候选面
- `data/province_boundaries.geojson`：每段只绘制一次的共享省界中线；穿过湖泊的省界采用直线
- `data/national_outline.geojson`：国土外轮廓候选
- `data/national_border_land.geojson`：裁掉海岸线重叠部分后的国界线
- `data/hydrology_lines.geojson`：河流候选折线
- `data/canals.geojson`：两段南部运河中心线及尺刻状符号
- `data/seasonal_rivers.geojson`：原图中邻国蓝色虚线季节性河流
- `data/cross_border_rivers.geojson`：从原图补充提取的跨境河流线段
- `data/neighbor_borders.geojson`：从原图筛出的邻国国界候选；排除邻国内部浅灰省界残线，蓝色水线单独归入河流图层
- `data/coastlines.geojson`：根据官方高清无河流图分块提取的海岸线、岛岸与河口岸线；每块 1024×1024 像素，使用无文字图交叉核对文字伪影，并剔除实心城市点；闭合岛岸从拼接后的海面边界补齐
- `data/ocean_surface_path.json`：与岸线来自同一批小块的高清海面填充路径；仅海面填充在块边缘增加 1.5 个源图像素的重叠，以封住扫描接缝，岛屿描边仍使用细节轮廓
- `data/ocean_water.geojson`：从原图海水颜色连通区域矢量化的海域
- `data/lakes.geojson`：湖泊候选面
- `data/settlement_markers.geojson`：城市（黑点）与省会（红点）标记候选
- `data/region_code_labels.geojson`：行政区代码标注点，由行政区合并面几何中心计算，中心落在面外时取面内点
- `data/region_codes.csv`：代码与组件关联表
- `data/ocr_candidates.csv`：OCR 文字候选及置信度、像素外接框
- `data/map_metadata.json`：地图标题、版本和坐标说明

## 可复现脚本
`extract_map_features.py` 可从输入图片重新生成行政区面、国土轮廓、水系线、湖泊、点标记、代码标注、OCR 候选和诊断掩膜。需要 Python、OpenCV、NumPy、Pandas、Pillow、Shapely、scikit-image、pytesseract 和 Tesseract 中文/英文语言包。

`extract_context_layers.py` 从基础原图提取海域、邻国国界和跨境河流候选图层；邻国边界筛选会排除浅灰内部线与蓝色水系笔画。它也保留早期海岸线提取流程；正式海岸线请使用下方的高清分块工具。
```bash
python extract_context_layers.py
```

`tools/extract_official_coastline_tiled.py` 逐块处理 `official_map` 中的高清无河流图和无文字图，以 48 像素重叠拼接相邻块；不会建立整张高清图的海水蒙版。它同时生成海面填充和岸线数据，过滤块边缘的人工接缝，并从闭合海面边界补齐岛岸。可指定诊断目录，保存各小块的岸线叠加图供人工复核：
```bash
python tools/extract_official_coastline_tiled.py --source-dir <official_map目录> --diagnostics-dir <分块复核目录>
```

`build_administrative_boundaries.py` 从行政区面与国土轮廓重新生成共享省界中线，并裁掉国界、省界与细致海岸线重叠的片段；同时按行政区合并面的几何中心生成半透明代码标注，再更新 Canvas 页面和独立 SVG 预览。生成流程需要 SciPy；目前根据原图将 2、3、4、5、6、10 号湖泊判为界湖并拉直穿湖省界。

`extract_hydrology_styles.py` 按原图蓝色虚线重新描绘两条邻国季节性河流，并将南部两段尺刻状水道作为独立运河图层；两类水道都从普通河流层剔除，东南沿海河口网保留在河流层。运河中心线按高清原图描绘，刻度依据各处局部切线生成垂线。省份代码按对应面几何中心放置；`SW` 有两块分离区域，代码只标在面积较大的那块。

运行示例：
```bash
python extract_map_features.py input_map.png --out .
python extract_context_layers.py
python build_administrative_boundaries.py
python extract_hydrology_styles.py
```

## 重要说明
- 预览由矢量路径、点和文字独立绘制，不使用原始位图作为显示底图。
- 坐标是原图像素坐标，不是经纬度。
- 自动提取结果是候选数据，不是已完成测绘或法律边界数据。
- 行政区代码已做视觉辅助转录，仍应逐项核验。



