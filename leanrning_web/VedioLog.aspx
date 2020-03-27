
<%@ Page Language="C#" AutoEventWireup="true" CodeFile="VedioLog.aspx.cs" Inherits="_Default" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
   
    <title></title>
</head>
<body>
    <form id="form1" runat="server" method="post">
    <div>
  

 <div id="player"> </div>
          <textarea runat="server" id="showbox" rows="30" cols="80"></textarea>
        <div>
            <asp:Button ID="btn" runat="server" Text="log" OnClick="btn_Click" />
        </div>
        
        
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
          height: '390',
          width: '640',
          videoId: 'pVeq-0dIqpk',
          events: {
            'onReady': onPlayerReady,
            'onStateChange': onPlayerStateChange,
            'onPlaybackRateChange': onPlaybackRateChange
          }
        });
      }

      // 4. The API will call this function when the video player is ready.
      function onPlayerReady(event) {
          
          event.target.playVideo();
          t.value +="當前速度" +player.getPlaybackRate()+"\r\n";
         
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

           t.value += "更換速度"+player.getPlaybackRate()+"\r\n";
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
