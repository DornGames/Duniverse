# 海岸线像素审阅工具

这套工具用于交叉核对已有海岸线矢量和原始大图，不会自动提取、补画或平滑海岸线。像素两侧的颜色只用于筛查，最终判定仍要看原图；海上小点可能是真岛，不能因为它两侧都是海水就自动删掉。

从项目根目录运行：

```powershell
python tools/coastline_pixel_audit.py
python -m http.server 8765
```

然后打开 `http://127.0.0.1:8765/tools/coastline_review.html`。页面支持放大、平移、筛选疑似线段、查看原图像素，以及记录和导出人工判定。

人工确认的误线会在 `data/coastlines.geojson` 中标记 `review_status: rejected` 和 `review_note`。地图重建时会跳过这些线，但保留原候选和判定理由，方便回看。
