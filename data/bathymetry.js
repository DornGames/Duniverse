(function(){
  'use strict';

  const MAX_DEPTH_M=15000;
  const MAP_PIXEL_KM=7.7719530772698535;
  const COLOR_STOPS=[
    [0,187,219,235],[200,164,207,229],[1000,137,195,224],[2500,114,181,216],
    [4500,93,169,214],[6000,79,150,204],[8000,71,133,190],[10000,65,118,176],
    [12000,58,104,160],[14000,51,92,145],[15000,45,82,135]
  ];

  // Coast-near outlets were matched to the mapped river network; the fans widen seaward.
  const RIVER_MOUTH_FANS=[
    {x:1448,y:1252,dx:.93,dy:.37,length:72,nearWidth:3,farWidth:26,relief:1150},
    {x:1328,y:435,dx:1,dy:.08,length:54,nearWidth:3,farWidth:20,relief:800},
    {x:1290,y:976,dx:1,dy:0,length:52,nearWidth:3,farWidth:22,relief:900},
    {x:1381,y:742,dx:1,dy:0,length:48,nearWidth:3,farWidth:18,relief:700}
  ].map(fan=>{
    const length=Math.hypot(fan.dx,fan.dy)||1;
    return{...fan,dx:fan.dx/length,dy:fan.dy/length};
  });

  // The deepest trough follows the ocean-facing side of the offshore island arc.
  const TRENCH_POINTS=[
    {x:1640,y:385},{x:1690,y:480},{x:1750,y:550},
    {x:1830,y:595},{x:1900,y:625},{x:1920,y:690},{x:1910,y:760},{x:1865,y:845}
  ];
  const TRENCH_SEGMENTS=makeSegments(TRENCH_POINTS);
  // The mapped island groups form a nearshore north-south belt and a separate offshore NW-SE arc.
  const COASTAL_BANK_SEGMENTS=makeSegments([
    {x:1278,y:460},{x:1298,y:498},{x:1308,y:535},{x:1325,y:680},
    {x:1338,y:830},{x:1351,y:975},{x:1375,y:1055}
  ]);
  const ARC_BRANCH_SEGMENTS=makeSegments([
    {x:1742,y:658},{x:1744,y:684},{x:1745,y:710},{x:1750,y:760},{x:1775,y:805}
  ]);
  const ARCHIPELAGO_PLATFORM_SEGMENTS=makeSegments([
    {x:1728,y:661},{x:1744,y:674},{x:1760,y:683},{x:1785,y:681},
    {x:1808,y:686},{x:1829,y:695}
  ]);
  // A submerged bank extends from the continental margin toward the northern island group.
  const SHELF_SPUR_SEGMENTS=makeSegments([
    {x:1348,y:470},{x:1410,y:461},{x:1470,y:448},{x:1530,y:438},{x:1571,y:436},{x:1602,y:448}
  ]);
  // A broad shelf wedge eases the transition into the northern island bank.
  const NORTHERN_SHELF_WEDGE=[{x:1400,y:200},{x:1600,y:420},{x:1360,y:470}].map((point,index,points)=>{
    const next=points[(index+1)%points.length],dx=next.x-point.x,dy=next.y-point.y;
    return{x:point.x,y:point.y,dx,dy,length:Math.hypot(dx,dy)};
  });
  const BASIN_AXIS=makeSegments([
    {x:1475,y:510},{x:1520,y:565},{x:1570,y:635},{x:1600,y:715},{x:1608,y:805},{x:1590,y:860}
  ]);
  // The southern island chain closes the basin with two offset submarine ridge sections.
  const SOUTHERN_RIDGE_EAST=makeSegments([
    {x:1745,y:710},{x:1768,y:750},{x:1784,y:785},{x:1750,y:812},
    {x:1722,y:845},{x:1708,y:870}
  ]);
  const SOUTHERN_RIDGE_WEST=makeSegments([
    {x:1690,y:875},{x:1650,y:883},{x:1595,y:881},
    {x:1550,y:895},{x:1509,y:918},{x:1470,y:950}
  ]);
  // Offset spreading segments produce broad rises in the outer ocean rather than one continuous stripe.
  const OUTER_RIDGE_NORTH=makeSegments([
    {x:2055,y:165},{x:2080,y:220},{x:2108,y:280},{x:2135,y:345},{x:2150,y:385}
  ]);
  const OUTER_RIDGE_SOUTH=makeSegments([
    {x:2170,y:460},{x:2190,y:520},{x:2215,y:580},{x:2245,y:650},{x:2255,y:705}
  ]);

  function makeSegments(points){
    const sampled=[points[0]];
    for(let i=0;i<points.length-1;i++){
      const a=points[Math.max(0,i-1)],b=points[i],c=points[i+1],d=points[Math.min(points.length-1,i+2)];
      for(let j=1;j<=3;j++){
        const t=j/3,t2=t*t,t3=t2*t;
        const interpolate=(p0,p1,p2,p3)=>.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t2+(-p0+3*p1-3*p2+p3)*t3);
        sampled.push({x:interpolate(a.x,b.x,c.x,d.x),y:interpolate(a.y,b.y,c.y,d.y)});
      }
    }
    return sampled.slice(1).map((point,index)=>{
      const start=sampled[index],dx=point.x-start.x,dy=point.y-start.y;
      return{ax:start.x,ay:start.y,dx,dy,lengthSquared:dx*dx+dy*dy};
    });
  }

  function makeColorLookup(){
    const lut=new Uint8ClampedArray((MAX_DEPTH_M+1)*4);
    let stopIndex=0;
    for(let depth=0;depth<=MAX_DEPTH_M;depth++){
      while(stopIndex<COLOR_STOPS.length-2&&depth>COLOR_STOPS[stopIndex+1][0])stopIndex++;
      const low=COLOR_STOPS[stopIndex],high=COLOR_STOPS[stopIndex+1];
      const t=Math.max(0,Math.min(1,(depth-low[0])/(high[0]-low[0])));
      const offset=depth*4;
      lut[offset]=Math.round(low[1]+(high[1]-low[1])*t);
      lut[offset+1]=Math.round(low[2]+(high[2]-low[2])*t);
      lut[offset+2]=Math.round(low[3]+(high[3]-low[3])*t);
      lut[offset+3]=255;
    }
    return lut;
  }
  const COLOR_LOOKUP=makeColorLookup();

  function smoothStep(value){
    const t=Math.max(0,Math.min(1,value));
    return t*t*(3-2*t);
  }

  function shelfWedgeWeight(x,y){
    if(x<1325||x>1635||y<165||y>505)return 0;
    let insideDistance=Infinity;
    for(const edge of NORTHERN_SHELF_WEDGE){
      const distance=(edge.dx*(y-edge.y)-edge.dy*(x-edge.x))/edge.length;
      insideDistance=Math.min(insideDistance,distance);
    }
    return smoothStep((insideDistance+25)/65);
  }

  function latticeValue(x,y,seed){
    let n=(Math.imul(x,374761393)+Math.imul(y,668265263)+seed)|0;
    n=Math.imul(n^(n>>>13),1274126177);
    return ((n^(n>>>16))>>>0)/2147483647.5-1;
  }
  function valueNoise(x,y,scale,seed){
    const gx=x/scale,gy=y/scale,ix=Math.floor(gx),iy=Math.floor(gy);
    const tx=smoothStep(gx-ix),ty=smoothStep(gy-iy);
    const a=latticeValue(ix,iy,seed),b=latticeValue(ix+1,iy,seed);
    const c=latticeValue(ix,iy+1,seed),d=latticeValue(ix+1,iy+1,seed);
    return (a+(b-a)*tx)*(1-ty)+(c+(d-c)*tx)*ty;
  }
  function buildReliefNoise(width,height){
    const step=4,w=Math.ceil(width/step)+1,h=Math.ceil(height/step)+1,data=new Float32Array(w*h);
    for(let y=0;y<h;y++)for(let x=0;x<w;x++){
      data[y*w+x]=220*valueNoise(x*step,y*step,80,1947)+85*valueNoise(x*step,y*step,32,8123);
    }
    return{step,width:w,data};
  }
  function sampleReliefNoise(noise,x,y){
    const gx=x/noise.step,gy=y/noise.step,ix=Math.floor(gx),iy=Math.floor(gy);
    const tx=gx-ix,ty=gy-iy,i=iy*noise.width+ix,d=noise.data;
    return (d[i]+(d[i+1]-d[i])*tx)*(1-ty)+(d[i+noise.width]+(d[i+noise.width+1]-d[i+noise.width])*tx)*ty;
  }

  // Shelf width varies along the mapped coast: broad sedimentary margins and narrower headlands.
  const SHELF_WIDTHS=[[0,410],[240,520],[435,820],[580,650],[742,880],
    [880,520],[976,780],[1100,720],[1252,1050],[1448,850]];
  function shelfWidthAt(y){
    for(let i=1;i<SHELF_WIDTHS.length;i++){
      const a=SHELF_WIDTHS[i-1],b=SHELF_WIDTHS[i];
      if(y<=b[0])return a[1]+(b[1]-a[1])*smoothStep((y-a[0])/(b[0]-a[0]));
    }
    return SHELF_WIDTHS[SHELF_WIDTHS.length-1][1];
  }
  function continentalProfile(distanceKm,shelfWidth){
    if(distanceKm<=shelfWidth)return 180*smoothStep(distanceKm/shelfWidth);
    const beyond=distanceKm-shelfWidth;
    if(beyond<=260)return 180+3020*smoothStep(beyond/260);
    if(beyond<=600)return 3200+2000*smoothStep((beyond-260)/340);
    if(beyond<=900)return 5200+600*smoothStep((beyond-600)/300);
    return 5800;
  }

  function mainlandCoastByRow(water,width,height,sourceWidth){
    const coast=new Float32Array(height),smoothed=new Float32Array(height);
    let previous=0;
    for(let y=0;y<height;y++){
      let x=Math.min(sourceWidth-1,width-1),row=y*width;
      while(x>0&&water[row+x])x--;
      if(x>0)previous=x;
      coast[y]=previous;
    }
    // Remove pixel-scale wiggles from the shelf break while preserving broad bays and headlands.
    for(let y=0;y<height;y++){
      let sum=0,weight=0;
      for(let dy=-24;dy<=24;dy++){
        const yy=Math.max(0,Math.min(height-1,y+dy)),w=25-Math.abs(dy);
        sum+=coast[yy]*w;weight+=w;
      }
      smoothed[y]=sum/weight;
    }
    return smoothed;
  }

  function distanceSquaredToPolyline(x,y,segments){
    let nearest=Infinity;
    for(const segment of segments){
      const {ax,ay,dx,dy,lengthSquared}=segment;
      const t=lengthSquared?Math.max(0,Math.min(1,((x-ax)*dx+(y-ay)*dy)/lengthSquared)):0;
      const offsetX=x-(ax+t*dx),offsetY=y-(ay+t*dy);
      nearest=Math.min(nearest,offsetX*offsetX+offsetY*offsetY);
    }
    return nearest;
  }

  function riverFanRelief(x,y){
    let relief=0;
    for(const fan of RIVER_MOUTH_FANS){
      const offsetX=x-fan.x,offsetY=y-fan.y;
      const along=offsetX*fan.dx+offsetY*fan.dy;
      if(along<=0||along>=fan.length)continue;
      const across=-offsetX*fan.dy+offsetY*fan.dx;
      const progress=along/fan.length;
      const width=fan.nearWidth+(fan.farWidth-fan.nearWidth)*progress;
      const crossFactor=Math.exp(-.5*(across/width)**2);
      relief+=fan.relief*Math.sin(Math.PI*progress)*crossFactor;
    }
    return relief;
  }

  function buildDistanceWaterMask(maskPixels,width,height){
    const size=width*height,rgba=maskPixels.data;
    const water=new Uint8Array(size),exterior=new Uint8Array(size),queue=new Int32Array(size);
    for(let i=0;i<size;i++)water[i]=rgba[i*4+3]>0?1:0;

    let head=0,tail=0;
    const enqueueExterior=index=>{
      if(!water[index]&&!exterior[index]){exterior[index]=1;queue[tail++]=index;}
    };
    for(let x=0;x<width;x++){
      enqueueExterior(x);
      enqueueExterior((height-1)*width+x);
    }
    for(let y=0;y<height;y++){
      enqueueExterior(y*width);
      enqueueExterior(y*width+width-1);
    }
    while(head<tail){
      const index=queue[head++],x=index%width;
      if(x>0)enqueueExterior(index-1);
      if(x+1<width)enqueueExterior(index+1);
      if(index>=width)enqueueExterior(index-width);
      if(index+width<size)enqueueExterior(index+width);
    }

    // Enclosed land holes are islands: keep them on the rendered mask, but let the continental shelf flow beneath them.
    for(let i=0;i<size;i++)if(!water[i]&&!exterior[i])water[i]=1;
    return water;
  }

  function distanceTransform1D(input,length,output,vertices,boundaries){
    let first=-1;
    for(let i=0;i<length;i++){
      if(Number.isFinite(input[i])){first=i;break;}
    }
    if(first<0){output.fill(Infinity,0,length);return;}

    let k=0;
    vertices[0]=first;
    boundaries[0]=-Infinity;
    boundaries[1]=Infinity;
    for(let q=first+1;q<length;q++){
      if(!Number.isFinite(input[q]))continue;
      let p=vertices[k];
      let crossing=((input[q]+q*q)-(input[p]+p*p))/(2*(q-p));
      while(crossing<=boundaries[k]){
        k--;
        p=vertices[k];
        crossing=((input[q]+q*q)-(input[p]+p*p))/(2*(q-p));
      }
      k++;
      vertices[k]=q;
      boundaries[k]=crossing;
      boundaries[k+1]=Infinity;
    }

    k=0;
    for(let q=0;q<length;q++){
      while(boundaries[k+1]<q)k++;
      const delta=q-vertices[k];
      output[q]=delta*delta+input[vertices[k]];
    }
  }

  function distanceToLandSquared(waterMask,width,height){
    const size=width*height;
    const horizontal=new Float32Array(size);
    const distances=new Float32Array(size);
    const maxDimension=Math.max(width,height);
    const input=new Float64Array(maxDimension);
    const output=new Float64Array(maxDimension);
    const vertices=new Int32Array(maxDimension);
    const boundaries=new Float64Array(maxDimension+1);
    for(let y=0;y<height;y++){
      const row=y*width;
      for(let x=0;x<width;x++)input[x]=waterMask[row+x]?Infinity:0;
      distanceTransform1D(input,width,output,vertices,boundaries);
      for(let x=0;x<width;x++)horizontal[row+x]=output[x];
    }

    for(let x=0;x<width;x++){
      for(let y=0;y<height;y++)input[y]=horizontal[y*width+x];
      distanceTransform1D(input,height,output,vertices,boundaries);
      for(let y=0;y<height;y++)distances[y*width+x]=output[y];
    }
    return distances;
  }

  function createRaster({width,height,sourceWidth,oceanPath,islandPaths=[]}){
    if(!oceanPath||!width||!height)return null;
    const maskCanvas=document.createElement('canvas');
    maskCanvas.width=width;maskCanvas.height=height;
    const maskContext=maskCanvas.getContext('2d',{willReadFrequently:true});
    maskContext.fillStyle='#fff';
    maskContext.fill(oceanPath,'evenodd');
    maskContext.fillRect(Math.max(0,sourceWidth-2),0,width-sourceWidth+2,height);
    const maskPixels=maskContext.getImageData(0,0,width,height);
    const distanceWater=buildDistanceWaterMask(maskPixels,width,height);
    const distanceSquared=distanceToLandSquared(distanceWater,width,height);
    const islandCanvas=document.createElement('canvas');
    islandCanvas.width=width;islandCanvas.height=height;
    const islandContext=islandCanvas.getContext('2d',{willReadFrequently:true});
    islandContext.fillStyle='#fff';islandContext.strokeStyle='#fff';islandContext.lineWidth=14;
    islandContext.lineJoin='round';islandContext.lineCap='round';
    for(const path of islandPaths){islandContext.fill(path,'evenodd');islandContext.stroke(path);}
    const bankCanvas=document.createElement('canvas');bankCanvas.width=width;bankCanvas.height=height;
    const bankContext=bankCanvas.getContext('2d',{willReadFrequently:true});
    bankContext.filter='blur(8px)';bankContext.drawImage(islandCanvas,0,0);
    const bankPixels=bankContext.getImageData(0,0,width,height).data;
    const reliefNoise=buildReliefNoise(width,height);
    const mainlandCoast=mainlandCoastByRow(distanceWater,width,height,sourceWidth);
    const size=width*height;
    const outputCanvas=document.createElement('canvas');
    outputCanvas.width=width;outputCanvas.height=height;
    const outputContext=outputCanvas.getContext('2d');
    const image=outputContext.createImageData(width,height);
    const pixels=image.data,mask=maskPixels.data;
    const ridgeSegments=makeSegments([
      {x:1571,y:436},{x:1602,y:448},{x:1674,y:526},{x:1742,y:658},{x:1808,y:686}
    ]);

    for(let y=0;y<height;y++){
      const row=y*width,rowShelfWidth=shelfWidthAt(y);
      for(let x=0;x<width;x++){
        const index=row+x,alpha=mask[index*4+3];
        if(alpha<8)continue;
        const coastDistancePx=Math.sqrt(distanceSquared[index]);
        const seawardDistancePx=Math.max(0,x-mainlandCoast[y]);
        const marginDistance=Math.min(coastDistancePx,seawardDistancePx)*MAP_PIXEL_KM;
        let depth=continentalProfile(marginDistance,rowShelfWidth);

        const basinDistanceSquared=distanceSquaredToPolyline(x,y,BASIN_AXIS);
        const basinGate=smoothStep((marginDistance-rowShelfWidth*.55)/450);
        depth+=1650*basinGate*Math.exp(-.5*basinDistanceSquared/(68*68));
        depth+=sampleReliefNoise(reliefNoise,x,y)*smoothStep((depth-400)/1800);

        const ridgeDistanceSquared=distanceSquaredToPolyline(x,y,ridgeSegments);
        const mainArc=2800*Math.exp(-.5*ridgeDistanceSquared/(24*24));
        const coastalBankDistanceSquared=distanceSquaredToPolyline(x,y,COASTAL_BANK_SEGMENTS);
        depth-=900*Math.exp(-.5*coastalBankDistanceSquared/(12*12));
        const arcBranchDistanceSquared=distanceSquaredToPolyline(x,y,ARC_BRANCH_SEGMENTS);
        const branchFade=1-smoothStep((y-710)/95);
        const branchArc=2200*branchFade*Math.exp(-.5*arcBranchDistanceSquared/(21*21));
        const southernEastDistance=distanceSquaredToPolyline(x,y,SOUTHERN_RIDGE_EAST);
        const southernWestDistance=distanceSquaredToPolyline(x,y,SOUTHERN_RIDGE_WEST);
        const eastFade=smoothStep((y-700)/75)*smoothStep((915-y)/75);
        const westFade=smoothStep((x-1440)/70)*smoothStep((1740-x)/70);
        const easternRidge=2800*eastFade*Math.exp(-.5*southernEastDistance/(25*25));
        const westernRidge=2500*westFade*Math.exp(-.5*southernWestDistance/(27*27));
        depth-=Math.max(mainArc,branchArc,easternRidge,westernRidge);
        const ridgeNorth=distanceSquaredToPolyline(x,y,OUTER_RIDGE_NORTH);
        const ridgeSouth=distanceSquaredToPolyline(x,y,OUTER_RIDGE_SOUTH);
        const northFade=smoothStep((y-165)/100)*smoothStep((385-y)/100);
        const southFade=smoothStep((y-460)/110)*smoothStep((705-y)/110);
        depth-=700*northFade*Math.exp(-.5*ridgeNorth/(35*35));
        depth-=600*southFade*Math.exp(-.5*ridgeSouth/(38*38));
        depth-=riverFanRelief(x,y);

        // Nearby islands merge into shared banks; scattered tiny islets make only subtle rises.
        const bank=smoothStep((bankPixels[index*4+3]/255-.30)/.55);
        depth+=(Math.min(depth,500)-depth)*bank;
        const spurDistance=Math.sqrt(distanceSquaredToPolyline(x,y,SHELF_SPUR_SEGMENTS));
        const spur=smoothStep((62-spurDistance)/42);
        depth+=(Math.min(depth,450)-depth)*spur;
        const shelfWedge=shelfWedgeWeight(x,y);
        depth+=(Math.min(depth,260)-depth)*shelfWedge;

        // A shared shallow island platform surrounds the group; its sheltered inner water is flatter and shallower.
        const platformDistance=Math.sqrt(distanceSquaredToPolyline(x,y,ARCHIPELAGO_PLATFORM_SEGMENTS));
        const platform=smoothStep((56-platformDistance)/30);
        depth+=(Math.min(depth,260)-depth)*platform;
        const lagoonX=(x-1775)/34,lagoonY=(y-685)/22;
        const lagoon=smoothStep((1-Math.hypot(lagoonX,lagoonY))/.45);
        depth+=(Math.min(depth,45)-depth)*lagoon;

        const trenchDistanceSquared=distanceSquaredToPolyline(x,y,TRENCH_SEGMENTS);
        const trenchEnds=smoothStep((y-385)/155)*smoothStep((845-y)/155);
        const trenchWidth=15+5*smoothStep((y-540)/170);
        const deepest=6200*Math.exp(-.5*((y-690)/70)**2);
        depth+=(3200+deepest)*trenchEnds*Math.exp(-.5*trenchDistanceSquared/(trenchWidth*trenchWidth));
        depth=Math.max(0,Math.min(MAX_DEPTH_M,depth));
        const colorOffset=Math.round(depth)*4,pixelOffset=index*4;
        pixels[pixelOffset]=COLOR_LOOKUP[colorOffset];
        pixels[pixelOffset+1]=COLOR_LOOKUP[colorOffset+1];
        pixels[pixelOffset+2]=COLOR_LOOKUP[colorOffset+2];
        pixels[pixelOffset+3]=alpha;
      }
    }

    outputContext.putImageData(image,0,0);
    return outputCanvas;
  }

  window.DUNIVERSE_BATHYMETRY={MAX_DEPTH_M,createRaster};
})();
