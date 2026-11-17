window.DUNIVERSE_HIGH_SPEED_RAIL = {
  coordinateSystem: 'source image pixels; same local map as D1 and D2',
  placementRule: 'each station is midway between the matching cardinal points of the D1 and D2 highway rings',
  routes: [
    {
      id: 'G1',
      name: '臭都高铁环线',
      color: '#247ca3',
      closed: true,
      points: [
        [1001, 856.5],
        [1012, 858.5],
        [1023, 864],
        [1031, 872],
        [1036.5, 885.5],
        [1033, 899],
        [1024, 910],
        [1011, 918],
        [1000, 920.5],
        [986, 916],
        [976, 907],
        [969, 895.5],
        [973, 883],
        [982, 871],
        [991, 861]
      ]
    }
  ],
  stations: [
    { id: 'choudu-north', name: '臭都北站', direction: 'north', x: 1001, y: 856.5 },
    { id: 'choudu-east', name: '臭都东站', direction: 'east', x: 1036.5, y: 885.5 },
    { id: 'choudu-south', name: '臭都南站', direction: 'south', x: 1000, y: 920.5 },
    { id: 'choudu-west', name: '臭都西站', direction: 'west', x: 969, y: 895.5 }
  ]
};
