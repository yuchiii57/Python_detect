<%@ Page Language="C#" AutoEventWireup="true" CodeFile="index.aspx.cs" Inherits="index" %>

<!DOCTYPE html>


<head runat="server">
    <meta name="viewport" content="width=device-width; initial-scale=1.0; maximum-scale=1.0; user-scalable=0;">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>Learning Player</title>
     <style type="text/css">
  body {
 font-family:'Gill Sans MT';
 font-weight:bold;
    background-color: #fff;
  
  }

  h2 {
    font-family: Comic Sans MS,arial,helvetica,sans-serif;}
  </style>
    
  <link href="../css/Style.css" type="text/css" rel="stylesheet" />
   
        <script src="js/OwlCarousel2-2.2.0/assets/owlcarousel/owl.carousel.js"></script>

</head>
<body>
<div class="container" >
  <div>
      <nav style="background-color:rgb(6,32,62);width:100%;border-bottom:1px solid #fff;font-family:'Gill Sans MT';">
          <ul class="ull"> 
         
              <li class="logo" style="text-align:left;"><img src="imgs/logo.png" style="float:left"   >

              </li>
          
              <li class="dropdownlist" style="color:#fff"> <a href="#pro" > Introduction</a></li>
              <li class="dropdownlist"> <a href="#idea" > Idea </a></li>
              <li class="dropdownlist">  <a href="#future" >Future  </a></li>
              <li class="dropdownlist">  <a href="#Team" >About Us </a> </li>
              <li class="dropdownlist"> <a href="Login.aspx" >Login|Register</a></>
          </ul>
      </nav>
  </div>
    <div class="coverflow">
     <div>
		<img src="imgs/catch your eyes.png"  style="    height: 100%;"/>
		
	</div>
	<div  class="coverflow">
		<img src="imgs/learn what you see see what you learn.png"  style="     height: 100%;" />
	
	</div>
	<div  class="coverflow">
		<img src="imgs/learning online.png"  style="  height: 100%;" />
		
	</div>
	<div  class="coverflow">
		<img src="imgs/the new generation of learning.png"   style="   height: 100%;"/>
	
	</div>
        <div  class="coverflow">
		<img src="imgs/turn.png"   style="   height: 100%;"/>
		
	</div>
	
</div>
    <div  id="pro">
             <div  style="margin-top:50px;height:300px;" >
                  <img src="imgs/FacialPayment02.jpg" style="float:left;"  /> 
                     <h3 style="text-align:center;padding-right:20px;padding-left:20px;padding-top:50px;"><span class="a2">Learning<br /> Online</span><br />
                         <a class="a1 tcontent">
                       提供學生與教師一個上傳教學影片及學習的系統平台。<br />本線上平台的特色是學習不受時間地點的限制，可以隨時隨地上課，<br />對於喜歡自主學習及自由利用時間的學習者相當便利。</h3>
                                </a>
                     </div>

   
   
     
            <div style="margin-top:50px;height:400px;">
                 <img src="imgs/face.png" style="float:right;" /> 
            <h3 style="text-align:center;padding-right:20px;padding-left:20px;padding-top:50px;"><span class="a2"> Attention <br /> Data</span><br/>
                <a class="a1 tcontent">
                 將學生上課專注度數據化，讓學生可以快速自我檢視；<br />而老師可以依數據作教學上的調整。
                    </a></h3>
</div>

  
             <div style="margin-top:50px; height:500px; margin-bottom:50px;">
                    <img src="imgs/emotion.gif" style="float:left;height:400px; "/>
              <h3 style="text-align:center;padding-right:20px;padding-left:20px;padding-top:100px;"><span class="a2">Emotion<br />Analysis</span><br />
                 
                   <a class="a1 tcontent">
                      分析及統計學生對課程的反應，
                     例如當老師講故事去引導理論、<br />或者以其他比喻來讓學生容易記憶等上課片段，<br />
                     老師可以參考該時間區段內學生學習的臉部表情分析結果，<br />
                     來推斷成效，學生是感興趣還是聽不懂？<br />或是覺得很有趣而會心一笑等。
                       </a></h3>
          </div>
        </div>
    <div id="idea"  style="background-color:rgb(6,32,62);padding:20px 100px 100px 100px;">
        <div class="text-center" style="text-align:center;padding-top:50px" >
      <h1 style="color:#fff">Idea</h1> 
            <img src="imgs/ideaicon.png" style="float:right;margin-top:-32px" />
     </div> 
    <div class="text-center" style="color:#fff;font-size:20px;font-family:'Microsoft JhengHei'">
       面對制式化的學習方式，在一間教室裡，老師播放投影幕上的教學內容，學生在台下坐著學習。身為學生的我們，希望能夠翻轉這樣的
     教學方式，不只是侷限於「一間教室內」學習，取而代之的是能夠「隨時隨地」的進行。在一間教室內，老師能夠使用肉眼分辨學生是否專心，但進行數位學習，並沒辦法確認學生的學習狀況，為了打破此隔閡，我們便錄影擷取學生的學習情況，辨別每個學生的「專心程度」進行分析。
     </div>

    </div>
    <div id="future"style="padding:100px 100px 50px 100px">

        <div class="text-center" style="text-align:center;" >
         <h1 >Future Application</h1>
     </div>
    <div class="text-center" style="margin-left:300px;color:#808080;font-size:20px;font-family:'Microsoft JhengHei'">
    
        <span style="color:red">教學系統 </span><br />
        以專注度分析來協助教師教課品質和學生的吸收程度來做教學上調整，也讓學生可以自我檢視<br /><br />

        <span style="color:red">車禍預防</span> <br />
        可偵測駕駛精神狀況，發出警示提醒駕駛是否找代駕或是休息再上路<br /><br />

          <span style="color:red">銷售輔助</span> <br />
        可得知顧客購買心理，用以適合的銷售手法或是適合的產品來提升業績<br /><br />

         <span style="color:red"> 活動成效</span>  <br />
         可以數據來了解或檢討參與者融入與否、主辦者成功與否<br /><br />

     </div>
    
          <div>
         <img src="imgs/rocket1.gif" />
     </div>

    </div>

       <div id="Team"style="padding:50px 100px 100px 100px;background-color:rgb(6,32,62)">

        <div class="text-center" style="text-align:center;color:#fff" >
         <h1 >Our Team</h1>
     </div>
    <div class="text-center" style="line-height:20px;color:#e13c3c;font-size:medium;font-family:'Microsoft JhengHei'">
        <div  >
            <img src="imgs/team.png" />
        </div>
          
     </div>
   


    </div>




 </div>
 







 <footer class="footer">
      <table>
            <tr>
               <td style="padding-left:550px; border-right:1px solid #fff"> VAI @ Visual Attention Identification </td>
               
              </tr>

      </table>
        </footer>
    

</body>
 