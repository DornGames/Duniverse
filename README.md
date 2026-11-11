# Duniverse

## 𣿅国交互式地图集

Duniverse 是一份交互式地图与地图数字化项目。它将《𣿅国行政区划图（2026 年第六版）》整理为可缩放的浏览器地图，并提供行政区、边界、水系、城市、海底深度和陆地地形等独立图层。

![𣿅国行政区划图矢量预览](vector_preview.svg)

## 打开地图

无需安装应用或下载依赖，直接用浏览器打开仓库中的 [`index.html`](index.html) 即可。也可以从项目目录运行一个本地静态服务器：

```bash
python -m http.server 8000
```

然后访问 <http://localhost:8000>。

## 地图操作

- “水系”图层中可单独开关海底深度，色标范围为 0 至 −15,000 米；放大时海底颜色逐渐淡出，近景海洋与河流颜色一致。
- 滚动鼠标滚轮缩放地图，拖动地图平移视图。
- 点击地图上的地理事物查看介绍；右键任意地图位置可查看地形模型海拔，点击海洋仍查看海底深度。支持鼠标滚轮缩放、触屏双指捏合和拖拽平移。
- 点击大区或省份，右栏会显示预先计算的平均海拔、最高点和最低点及其地图位置；统计读取固定数据，不会在页面打开时扫描整张高程栅格。
- “陆地地形”图层可开关地势晕渲和北湩高原官方分界线；地图没有绘制等高线。
- 城市与省界覆盖全图，可在“城市、首都与省会”图层组中单独开关；先从命名山脉骨架和高程面提取山脊，再结合河网、城市锚点划分市界，并按地势起伏加入固定种子的曲线细节。高峰可在靠近市界且避开城市标记时成为界山；放大后省界由同一市界图层继续显示。
- 地图叠加土星自定义坐标网格：纬度每 1°、经度每 10°一条浅色半透明细虚线；边缘刻度使用本项目的数字度数，不显示方向字母。自定义纬度等于真实行星纬度的六分之一。
- 在顶部搜索框输入中文名、𣿅语名、行政区编号或城市邮政区号，可搜索国家、大区、省份、城市、山峰和已命名河流；选择后地图会定位到对应对象。
- 搜索行政区或地名可定位并查看可用介绍；地图点击恢复为打开地理要素信息，城市介绍已恢复。
- 拖动地图时，只要相应文字图层保持开启，中文和𣿅语标签都会继续显示。
- 右侧图层可分别显示或隐藏行政区边界、大区边界、水系、城市标记和行政区名；行政区名分为“编号”和“中文”两个子图层。
- 宏观视图会显示大区双语名称，中文在上、𣿅语小字在下；名称保持固定屏幕尺寸，进入省级视图前逐渐淡出。
- 行政区编号和中文名称随放大同步淡入，继续放大后同步淡出；两个子图层仍可独立开关。
- 放大后地图会显示更多细节。海岸线填充在约 65% 的缩放进度达到最高几何精度；东部海岛在概览中保持可见，点状小岛数据见 [`data/eastern_island_dots.geojson`](data/eastern_island_dots.geojson)。

## 项目内容

- **陆地地形**：固定种子生成山地、高原、河谷、平原和盆地地势；以地图中 87 座已标注山峰为高度锚点，结合河网、海岸与城市点位建模。数据为明确标注的推演值，不是实测高程；北湩高原分界线按官方地图描绘。
- **区域海拔统计**：为 8 个大区和 45 个省份预先计算平均海拔、最高点及最低点；极值附地图像素位置，并与地形模型一起保存。
- **海底地形**：宽大陆架、岛链海脊和浅海台地围合海盆；固定种子的细微地形起伏在刷新后保持一致。
- **交互式地图**：Canvas 绘制行政区和地图要素；支持缩放、平移、区域定位与分组图层控制。
- **行政区标注**：两字母编号与对应中文名称并列显示，可分别开关。
- **矢量预览**：[`vector_preview.svg`](vector_preview.svg) 可直接在浏览器、Inkscape 或 Illustrator 中打开。
- **地图数据**：提取并整理了行政区面、共享省界、大区界、国境线、邻国边界、海岸线、河流、运河、湖泊、城市及省会标记。陆地地形模型和逐城、逐河自查数据见 [`data/mountain_elevation_model.md`](data/mountain_elevation_model.md) 与对应审计表。
- **地名搜索与介绍**：支持按中文名、𣿅语名、行政区编号和城市邮政区号动态搜索国家、大区、省份、城市、山峰、湖泊和已命名河流。行政区、城市、山峰与湖泊信息分别维护在 [`data/region_directory.js`](data/region_directory.js)、[`data/city_directory.js`](data/city_directory.js)、[`data/mountain_peaks.js`](data/mountain_peaks.js) 与 [`data/lake_names.js`](data/lake_names.js)。地图上未能确认位置或没有简介的地名仍可搜索，并显示“暂无”。
- **数据处理工具**：仓库包含地图要素提取、边界生成和水系样式处理脚本，便于继续修订和重新生成图层。

概览时显示大区界和大区双语名称，不显示省界；逐渐放大时大区名先淡出，随后省名和编号淡入，并切换为更细的虚线省界。继续放大后，大区界和省界按层级隐藏。边界线在缩放时保持屏幕粗细不变。海岸线不作为大区边界绘制。大区归属与𣿅语名称维护在 [`data/major_region_membership.json`](data/major_region_membership.json)，大区名称标注位置记录在 `data/major_region_labels.geojson`。

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| `index.html` | 浏览器交互地图与 Canvas 渲染逻辑 |
| `data/bathymetry.js` | 海底深度场与连续深度配色 |
| `data/relief_elevation.js`、`data/relief_elevation.png`、`data/relief_tint.png` | 陆地高程、右键采样图和地图晕渲；高程采样图在地图出现后异步载入 |
| `data/relief_model.js`、`data/relief_model_metadata.json` | 山峰、城市、地貌控制点及模型说明 |
| `data/ridge_network.geojson` | 全图山脊骨架、局部脊线及用于市界的高峰控制点 |
| `data/city_boundaries.js`、`data/city_boundaries.geojson` | 全图城市边界与省界；市界根据山脊、河流、城市锚点和地形曲折度划分 |
| `data/region_elevation_stats.js`、`data/region_elevation_stats.csv` | 大区、省份的预计算海拔统计及复核表 |
| `data/mountain_elevation_model.md` | 山脉分区、地形假设与水文核查 |
| `vector_preview.svg` | 独立矢量地图预览 |
| `input_map.png` | 基础地图图像 |
| `Administrative-Division-Map-of-Donh-2026.png` | 高清地图图像，用于细节核对和水系描绘 |
| `data/` | 行政区、水系、海岸、标记和名称等地图数据 |
| `extract_map_features.py` | 从基础图像提取地图要素及 OCR 候选 |
| `extract_context_layers.py` | 提取海域、邻国边界和跨境河流候选 |
| `build_administrative_boundaries.py` | 生成共享省界、大区界及大区名称标注 |
| `extract_hydrology_styles.py` | 处理河流、季节性河流和运河图层 |
| `tools/build_region_elevation_stats.py` | 预计算大区、省份的平均海拔与地形极值 |
| `tools/build_city_boundaries.py` | 全图提取山脊网络，再在省界内结合河流和城市点生成带曲线细节的市界，并将省界并入放大视图 |
| `tools/` | 高清分块海岸线处理工具 |
| `diagnostics/` | 数据提取和海岸线复核时生成的诊断材料 |

行政区中文名称和𣿅语名称分别维护在 [`data/province_names.json`](data/province_names.json) 与 [`data/province_tongue_names.json`](data/province_tongue_names.json)，标注位置维护在 `data/region_code_labels.geojson`。地图标签按“中文（编号）”和下一行𣿅语显示，中文与𣿅语图层可分别开关。目前地图包含 45 个两字母行政区编号；其中 `TT` 在地图上有两个标注位置。

## 重新生成数据

地图页面已经包含当前展示所需的数据，日常使用不需要 Python。仓库中的脚本用于修改或重新生成数据层：

```bash
python extract_map_features.py input_map.png --out .
python extract_context_layers.py
python tools/extract_official_coastline_tiled.py --source-dir "path/to/official_map"
python build_administrative_boundaries.py
python extract_hydrology_styles.py
python tools/build_coordinate_catalog.py
```

这些步骤应在对应源图和数据准备完成后按顺序运行。将示例路径替换为官方高清地图所在目录；该源图目录不包含在本仓库中。`extract_context_layers.py` 也会生成基础海岸线候选，精细海岸线使用分块工具生成。

提取流程依赖 Python 及相应图像和地理数据处理库，包括 OpenCV、NumPy、Pandas、Pillow、Shapely、SciPy 和 scikit-image。OCR 提取还需要安装 Tesseract，并提供中文、英文语言包。各工具的其他参数可查看脚本说明或运行 `--help`。

## 坐标与数据说明

海底深度依据海岸线和岛群位置生成：大陆架宽度沿海岸变化，岛链之间形成海盆，南侧岛链的海脊与大陆架相接。海底颜色连续过渡，海沟最深为 −15,000 米。海底地貌参考 [NOAA 海底地形介绍](https://www.noaa.gov/education/resource-collections/ocean-coasts/ocean-floor-features)。

原图没有比例尺或地理控制点，所以基础图层仍以图像像素存储：原点在左上角，横轴向右、纵轴向下。项目另建立了一套根据给定国土面积定标的土星坐标参考，参数见 [`data/georeference.json`](data/georeference.json)，测绘假设和公式见[《𣿅国地图测绘报告》](saturn-map-survey-report.md)。该坐标系以土星 1 bar 参考椭球为基准，臭都为 0°经线、地图纬度约 4.7833°N（对应真实行星地理纬度约 28.7°N）；真实纬度 ±90°按六分之一记为地图纬度 ±15°，经度仍为东向 360°一周。

可用 `tools/georeference_map.py` 把像素点或现有 GeoJSON 转为该项目坐标。批量结果写入 `data/georeferenced/`，并标记为自定义土星坐标；这些坐标不是 WGS 84 / EPSG:4326，也不是官方地图原有的 GIS 坐标。
山峰源数据位于 [`data/mountain_peaks.js`](data/mountain_peaks.js)，地形图使用单独的 [`data/relief_mountain_peaks.js`](data/relief_mountain_peaks.js) 保存分区和少量位置修正，不覆盖原峰名数据。山峰和海拔是两个默认开启的独立图层；高度值按来源区分用户提供值与模型推定值。地形建模方法见 [`data/mountain_elevation_model.md`](data/mountain_elevation_model.md)。
城市地形锚点和调整后的标注点位于 `data/relief_city_coordinates.js`；16 位全分辨率海拔栅格与河道掩膜位于 `data/relief_elevation.js`。固定随机种子为 20261108，可用 `python tools/build_relief_map.py` 重新生成地形数据与审计材料。湖泊名称来自官方地图标注；未标名的邻国湖泊以项目方位名收录，图层默认开启并支持搜索。
`tools/build_coordinate_catalog.py` 会生成 316 个城市的全精度坐标数据和 53 个省/大区的四至及方向极端城市数据，页面介绍统一显示两位小数。对应文件为 [`data/city_coordinates.js`](data/city_coordinates.js) 与 [`data/region_extrema.js`](data/region_extrema.js)。

各省面积按提取面要素的像素面积权重归一化到全国给定面积，大区面积由所属省求和；归一化因子与原始覆盖率记录在 [`data/georeference.json`](data/georeference.json) 和测绘报告中。

边界和标注位置来自原图提取、矢量化或人工复核，适用于地图展示和项目编辑，不应视为官方 GIS 边界数据。

中文行政区名称根据项目附带的《𣿅国政区清单》逐项核对。原图标题、版本、制作方和版权信息见 [`data/map_metadata.json`](data/map_metadata.json)。

## 技术方法参考

空间数据检查与可复现工作流参考：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner. *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065 (2026). [论文页面](https://arxiv.org/abs/2609.00065)。

## 许可

仓库中的代码按 [`LICENSE`](LICENSE) 所列 Apache License 2.0 发布。原版地图图像及其版权声明以 `data/map_metadata.json` 和图像文件中的标注为准。
