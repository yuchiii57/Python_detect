$(function (){
  'use strict';

  var randomScalingFactor = function(){ return Math.round(Math.random()*100)};
  var lineChartData = {
    labels : ['1','2','3','4','5','6','7'],
    datasets : [
      {
        label: 'Commision',
        backgroundColor: '	#1d9cc9',
        borderColor : 'rgba(220,220,220,1)',
        pointBackgroundColor: '#1d9cc9',
        pointBorderColor : '#fff',
        data : [45345,56344,23455,43566,33509,23874,56123]
      },
      
    ]
  }

  var ctx = document.getElementById('canvas-1');
  var chart = new Chart(ctx, {
    type: 'line',
    data: lineChartData,
    options: {
      responsive: true
    }
  });


  var randomScalingFactor = function(){ return Math.round(Math.random()*100)};
  var barChartData = {
    labels : ['pro1','pro2','pro3','pro4','pro5','pro6'],
    datasets : [
  
      {
          label: 'Performance',
          backgroundColor: '	#79c346',
        borderColor : 'rgba(151,187,205,0.8)',
        highlightFill: '#79c346',
        highlightStroke: '#79c346',
        data : [2,3,4,5.6,2,1]
      }
    ]
  }
  var ctx = document.getElementById('canvas-2');
  var chart = new Chart(ctx, {
    type: 'bar',
    data: barChartData,
    options: {
        responsive: true,
    }
  });



  var doughnutData = {
    labels: 
     ['Approve', 'Unreviewed', 'Reject']
    ,
    datasets: [{
      data: [69,14,7],
      backgroundColor: [
        '#FF6384',
        '#36A2EB',
        '#FFCE56'
      ],
      hoverBackgroundColor: [
        '#FF6384',
        '#36A2EB',
        '#FFCE56'
      ]
    }]
  };
  var ctx = document.getElementById('canvas-3');
  var chart = new Chart(ctx, {
    type: 'doughnut',
    data: doughnutData,
    options: {
      responsive: true
    }
  });


  var radarChartData = {
      labels: ['pro1', 'pro2', 'pro3', 'pro4', 'pro5', 'pro6'],
    datasets: [
     
      {
          label: 'Achieving rate',
          backgroundColor: '#ff5454',
        borderColor: 'rgba(151,187,205,1)',
        pointBackgroundColor: '#ff5454',
        pointBorderColor: '#fff',
        pointHighlightFill: '#fff',
        pointHighlightStroke: '#ff5454',
        data: [50,45,46,24,13,76]
      }
    ]
  };
  var ctx = document.getElementById('canvas-4');
  var chart = new Chart(ctx, {
    type: 'radar',
    data: radarChartData,
    options: {
      responsive: true
    }
  });


  var pieData = {
      labels: ['Approve','Unreviewed','Reject'],
    datasets: [{
      data: [69,14,7],
      backgroundColor: [
        '#FF6384',
        '#36A2EB',
        '#FFCE56'
      ],
      hoverBackgroundColor: [
        '#FF6384',
        '#36A2EB',
        '#FFCE56'
      ]
    }]
  };
  var ctx = document.getElementById('canvas-5');
  var chart = new Chart(ctx, {
    type: 'pie',
    data: pieData,
    options: {
      responsive: true
    }
  });


  var polarData = {
    datasets: [{
      data: [
        11,
        16,
        7,
        3,
        14
      ],
      backgroundColor: [
        '#FF6384',
        '#4BC0C0',
        '#FFCE56',
        '#E7E9ED',
        '#36A2EB'
      ],
      label: 'My dataset' // for legend
    }],
    labels: [
      'Red',
      'Green',
      'Yellow',
      'Grey',
      'Blue'
    ]
  };
  var ctx = document.getElementById('canvas-6');
  var chart = new Chart(ctx, {
    type: 'polarArea',
    data: polarData,
    options: {
      responsive: true
    }
  });
});
