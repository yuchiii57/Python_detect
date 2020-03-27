<%@ Page Language="C#" AutoEventWireup="true" CodeFile="Class_teacher.aspx.cs" Inherits="Class_teacher" MasterPageFile="~/MasterPage2.master"Title="class" %>



<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder2" runat="server">
      <link href="../css/Style.css" type="text/css" rel="stylesheet" />
    <div style="text-align:center;margin-left:20px;margin-right:20px;">
    <div style="margin-top:10px;">
        <input type="button" runat="server" value="上傳教材"  style="float:right" onclick="location.href = 'Default.aspx'"/>  
        <table ">
            <tr  style="background-color:rgb(247,196,54);height:40px;color:#fff">
                <td style="width:100px;">課號</td>
                 <td style="width:200px;">課程名稱</td>
                 
                 <td style="width:300px;">影片預覽 </td>
                <td style="width:100px;">選擇</td>

            </tr>
            <tr style="background-color:rgb(240,240,240);height:60px;">
                <td >1 </td>
                <td>How to build (and rebuild) trust | Frances Frei</td>
                 <td class="vedio.container"><iframe width="300" height="200" src="https://www.youtube.com/embed/pVeq-0dIqpk" allow="autoplay; encrypted-media" allowfullscreen  "></iframe>  </td>
                <td>
                    <input type="button" runat="server" value="開始上課" class="btn" onclick="location.href = 'Vedio1.aspx'"/>
                </td>
               
            </tr>
              <tr style="background-color:rgb(245,245,245);height:60px;">
                <td >2 </td>
                <td>The person you really need to marry | Tracy McMillan </td>
                   <td class="vedio.container"><iframe  width="300" height="200" src="https://www.youtube.com/embed/P3fIZuW9P_M" allow="autoplay; encrypted-media" allowfullscreen></iframe>  </td>
                <td >
                    <input type="button" runat="server" value="開始上課" class="btn" onclick="location.href = 'Vedio2.aspx'"/>
                </td>
                    
            </tr>
              <tr style="background-color:rgb(240,240,240);height:60px;">
               <td >3 </td>
                <td>The next step in nanotechnology | George Tulevski</td>
                  <td class="vedio.container"><iframe  width="300" height="200" src="https://www.youtube.com/embed/Ds_rzoyyfF0" allow="autoplay; encrypted-media" allowfullscreen></iframe>  </td>
                <td>
                    <input type="button" runat="server" value="開始上課" class="btn" onclick="location.href = 'Vedio3.aspx'"/>
                </td>
                
            </tr>
              <tr style="background-color:rgb(245,245,245);height:60px;">
               <td >4 </td>
                <td>微笑隱藏的力量 | Ron Gutman</td>
               <td class="vedio.container"><iframe width="300" height="200" src="https://www.youtube.com/embed/rAIgkoHvMZ8" allow="autoplay; encrypted-media" allowfullscreen></iframe>  </td>
                <td>
                     <input type="button" runat="server" value="開始上課" class="btn" onclick="location.href = 'Vedio4.aspx'"/>
                </td>
                     

            </tr>
        </table>
    </div>
        </div>
</asp:Content>
