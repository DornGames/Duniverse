# 数据、坐标与精度边界

本页帮助读者区分原图信息、图像提取结果、人工修订和模型推算。Duniverse 是虚构世界地图项目；图层看起来像 GIS 数据，并不意味着它们都来自实测或官方空间数据库。

## 数据从哪里来

| 数据 | 主要文件 | 来源与处理 | 使用时的边界 |
| --- | --- | --- | --- |
| 地图图像 | [`Administrative-Division-Map-of-Donh-2026.png`](../Administrative-Division-Map-of-Donh-2026.png)、[`input_map.png`](../input_map.png)、[`data/map_metadata.json`](../data/map_metadata.json) | 依据《𣿅国行政区划图（2026 年第六版）》开展数字化。 | 原图没有比例尺、经纬网或地理配准。 |
| 行政区、国境和海岸 | [`data/administrative_regions.geojson`](../data/administrative_regions.geojson)、`data/province_boundaries.geojson`、`data/national_outline.geojson`、`data/coastlines.geojson` | 从地图图像提取、描绘或人工修订。部分修订附审阅图和批次记录。 | 适用于项目地图，不是法定边界，也未证明每处都完成同等精度的复核。 |
| 河流、运河和湖泊 | `data/rivers.geojson`、`data/canals.geojson`、`data/lakes.geojson` 及关联数据 | 参照地图中的水系线、符号和标注整理。 | 自动提取候选可能需要人工判定；地图中的线宽、符号和图像分辨率会影响位置。 |
| 城市与名称 | `data/city_directory.js`、`data/region_directory.js`、`data/mountain_peaks.js`、`data/lake_names.js` | 使用地图标注和项目名录维护名称、位置与介绍。 | 缺少资料或位置尚未确认的条目会保留空缺，不代表地图上没有该对象。 |
| 陆地地形 | [`data/relief_model_metadata.json`](../data/relief_model_metadata.json)、[`data/mountain_elevation_model.md`](../data/mountain_elevation_model.md)、`data/relief_elevation.js` | 固定种子的模型参考山峰、河流、城市、海岸和地貌控制点生成。 | 海拔值是项目推演值，不是测高或现场观测结果。 |
| 海底深度 | `data/bathymetry.js` | 根据海岸、岛群和地图设定生成大陆架、海盆等水深表现。 | 水深是模型值，不是海底测深数据。 |
| 道路和里程 | `data/highways.js`、`data/highway_geography.js`、`data/high_speed_rail.js` | 依据地图资料维护路线和站点。 | 高速里程按项目当前地图投影比例估算。 |

## 坐标有两种口径

### 原图像素坐标

原始几何以图像像素记录：原点在左上角，横坐标向右增加，纵坐标向下增加。提取图层沿用这套坐标，所以它们不是经纬度。

### 项目自定义土星坐标

项目另外建立了坐标标识 `DUNIVERSE-SATURN-SCALED-LAT-2026`。该参考采用土星 1 bar 参考椭球和等积圆柱投影，并按给定国土面积定标。臭都特别市被设为经度锚点，自定义地图纬度等于真实行星地理纬度的六分之一。

这个坐标系用于呈现该虚构世界的地图。锚点、尺度和纬度设定属于项目制图假设，不是从原图恢复的官方位置，也不等同于 WGS 84、EPSG:4326 或土星的官方坐标系统。

转换参数见 [`data/georeference.json`](../data/georeference.json)，公式和选择理由见[地图测绘报告](../saturn-map-survey-report.md)。

## 如何看待计算结果

- 行政区面积按提取面要素的像素面积权重归一到给定的全国面积；大区面积由所属省份汇总。
- 平均海拔和地形极值来自预计算地形模型，不是独立测量值。
- 城市坐标、行政区四至和高速里程依赖项目设定的坐标参考。
- 海底颜色与深度用于地图表达，不应解释为真实海洋测量。

## 数据复核的范围

仓库提供特征提取报告、坐标参数、海岸线修正批次和水文复核表。这些材料能说明某些数据怎样生成、某些问题怎样处理；局部检查通过不代表所有图层已经全面认证。审阅时请核对原图、数据文件和对应报告。

## 版权与许可

根目录的 [LICENSE](../LICENSE) 为仓库代码声明 Apache License 2.0。原版地图图像的版权标注记录在 [`data/map_metadata.json`](../data/map_metadata.json)，其中写明 © 2021–2026 Dornhub。不要把代码许可自动套用于原版地图图像或其他来源资产。