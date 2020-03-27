<%@ Page Language="C#" AutoEventWireup="true" CodeFile="Vedio4.aspx.cs" Inherits="Vedio4" %>

<!DOCTYPE html>
<link href="vendors/css/flag-icon.min.css" rel="stylesheet"/>
  <link href="vendors/css/font-awesome.min.css" rel="stylesheet"/>
  <link href="vendors/css/simple-line-icons.min.css" rel="stylesheet"/>

 <link href="css/Style.css" rel="stylesheet"/>
  <link href="css/style1.css" rel="stylesheet"/>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>微笑隱藏的力量 | Ron Gutman</title>
</head>
  
<body class="app header-fixed sidebar-fixed aside-menu-fixed aside-menu-hidden">
    
    <header class="app-header navbar">
        <button class="navbar-toggler mobile-sidebar-toggler d-lg-none" type="button">
            <span class="navbar-toggler-icon"></span>
        </button>
        <a class="navbar-brand" href="#"  style="border-right:solid 1px #fff">
            <img style="height:40px;float:left " src="imgs/logo.png" />
            <span style="color:#fff;height:40px;padding-left:10px"><b>VAI</b></span>
          
          
        </a>
        <button class="navbar-toggler sidebar-toggler d-md-down-none h-100 b-r-1 px-3" type="button">
            <span class="navbar-toggler-icon"></span>
        </button>
        <ul class="nav navbar-nav d-md-down-none mr-auto">
            <form class="form-inline px-4">
                <i class="fa fa-search"></i>
                <input class="form-control" type="text" placeholder="Search..."/>
            </form>
        </ul>
        <ul class="nav navbar-nav ml-auto">
            
          
              <li>
                 <a href="Login.aspx"> <img src="imgs/logout.png" /></a> 
              </li>
            </ul>
</header>
    
    


    <form id="form1" runat="server" method="post">
    <div>
    <div class="v_title">微笑隱藏的力量 | Ron Gutman</div>

 <div class="text-center" id="player" style="margin-left:300px;"  >  </div>   <%--youtube影片--%>
        <div style="float:right;margin-top:50px;margin-left:50px;" >
           <input type="button"  id="focus" runat="server" value="查看數據" OnClick="data_click"  style="width:150px;color:#000;margin-right:30px" />
           
            <input type="button"  id="End" runat="server" value="上課結束" onclick="location.href = 'Class.aspx'"  style="width:150px;color:#000;margin-right:30px" />
        </div>
          <textarea runat="server" id="showbox" rows="30" cols="80" style="display:none"></textarea>
               
        
    <script>        //yotubeapi

      // 2. This code loads the IFrame Player API code asynchronously.
      var tag = document.createElement('script');

      tag.src = "https://www.youtube.com/iframe_api";
      var firstScriptTag = document.getElementsByTagName('script')[0];
      firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

      // 3. This function creates an <iframe> (and YouTube player)
      //    after the API code downloads.
      var player;
      function onYouTubeIframeAPIReady() {
        player = new YT.Player('player', {
          height: '450',
          width: '800',
          videoId: 'rAIgkoHvMZ8',
          events: {
            'onReady': onPlayerReady,
            'onStateChange': onPlayerStateChange,
            'onPlaybackRateChange': onPlaybackRateChange
          }
        });
      }

      // 4. The API will call this function when the video player is ready.
      function onPlayerReady(event) {
          var number;
          var minutes;
          var time;
          var NowDate = new Date();
          var h = NowDate.getHours();
          var m = NowDate.getMinutes();
          var s = NowDate.getSeconds();
       
          number = player.getCurrentTime();
          minutes = number;
          time = h + ":" + m + ":" + s;
          event.target.playVideo();
          t.value += "當前速度" + player.getPlaybackRate() + minutes.toFixed(2) + "/" + time + "\r\n";
         
      }

      // 5. The API calls this function when the player's state changes.
      //    The function indicates that when playing a video (state=1),
      //    the player should play for six seconds and then stop.
      var done = false;
      var t = document.getElementById('showbox');
       var NowDate = new Date();
       function onPlayerStateChange(event) {
        
           if (event.data == YT.PlayerState.PAUSED ) {
               var number;
               var minutes;
               var time;
               var NowDate = new Date();
               var h = NowDate.getHours();
               var m = NowDate.getMinutes();
               var s = NowDate.getSeconds();
               t.value += "Pressed --->";
               number = player.getCurrentTime();
               minutes = number;
               time = h + ":" + m + ":" + s;
               t.value += minutes.toFixed(2)+"/" +time+ "\r\n";
               
           }
           if (event.data == 1) {
               var number;
               var minutes;
               var time;
               var NowDate = new Date();
               var h = NowDate.getHours();
               var m = NowDate.getMinutes();
               var s = NowDate.getSeconds();
               t.value += "Start --->";
               number = player.getCurrentTime();
               minutes = number;
               time = h + ":" + m + ":" + s;
               t.value += minutes.toFixed(2)+"/" +time + "\r\n";
           
              

           }
          
         
           
           
       }
       function onPlaybackRateChange(event) {
           var number;
           var minutes;
           var time;
           var NowDate = new Date();
           var h = NowDate.getHours();
           var m = NowDate.getMinutes();
           var s = NowDate.getSeconds();
          
           number = player.getCurrentTime();
           minutes = number;
           time = h + ":" + m + ":" + s;

           t.value += "更換速度" + player.getPlaybackRate() + minutes.toFixed(2) + "/" + time + "\r\n";
       }
       

      function stopVideo() {
       
        player.stopVideo();
      }</script>
        <script>
            //txt
            function btn_Click() {
                var fso =new ActiveXObject(Scripting.FileSystemObject);
                var f = fso.createtextfile("C:\Web\learning_web\Log\a.txt", 2, true);
                var dt=NowDate();
                var t = document.getElementById("showbox").innerText;

                f.writeLine(dt+t);
                f.close();
            }

        </script>
    </div>
    </form>
</body>
</html>
 

