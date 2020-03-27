<%@ Page Language="C#" Title="Class" AutoEventWireup="true" MasterPageFile="~/MasterPage.master" CodeFile="Class.aspx.cs" Inherits="Class1" %>

<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder2" Runat="Server">
      <link href="../css/Style.css" type="text/css" rel="stylesheet" />
    <form runat="server" id="form1">
    <div style="text-align:center;margin-left:20px;margin-right:20px;">
        

    <div style="margin-top:10px;">
        <table ">
            <tr  style="background-color:rgb(247,196,54);height:40px;color:#fff">
                <td style="width:100px;">課號</td>
                 <td style="width:200px;">課程名稱</td>
                 
                 <td style="width:300px;">影片預覽 </td>
                <td style="width:100px;">選擇</td>

            </tr>
            <tr style="background-color:rgb(240,240,240);height:60px;">
                <td >1 </td>
                <td><asp:label ID="label1" runat="server">How to build (and rebuild) trust | Frances Frei</asp:label></td>
                 <td class="vedio.container"><iframe width="300" height="200" src="https://www.youtube.com/embed/pVeq-0dIqpk" allow="autoplay; encrypted-media" allowfullscreen  "></iframe>  </td>
                <td>
                    <asp:button runat="server" Text="開始上課" class="btn" OnClick="C_click"/>
                </td>
               
            </tr>
              <tr style="background-color:rgb(245,245,245);height:60px;">
                <td >2 </td>
                <td><asp:label ID="label2" runat="server">The person you really need to marry | Tracy McMillan</asp:label> </td>
                   <td class="vedio.container"><iframe  width="300" height="200" src="https://www.youtube.com/embed/P3fIZuW9P_M" allow="autoplay; encrypted-media" allowfullscreen></iframe>  </td>
                <td >
                      <asp:button runat="server" Text="開始上課" class="btn" OnClick="C_click2"/>
                </td>
                    
            </tr>
              <tr style="background-color:rgb(240,240,240);height:60px;">
               <td >3 </td>
                <td><asp:label ID="label3" runat="server">The next step in nanotechnology | George Tulevski</asp:label></td>
                  <td class="vedio.container"><iframe  width="300" height="200" src="https://www.youtube.com/embed/Ds_rzoyyfF0" allow="autoplay; encrypted-media" allowfullscreen></iframe>  </td>
                <td>
                     <asp:button runat="server" Text="開始上課" class="btn" OnClick="C_click3"/>
                </td>
                
            </tr>
              <tr style="background-color:rgb(245,245,245);height:60px;">
               <td >4 </td>
                <td>微笑隱藏的力量 | Ron Gutman</td>
               <td class="vedio.container"><iframe width="300" height="200" src="https://www.youtube.com/embed/rAIgkoHvMZ8" allow="autoplay; encrypted-media" allowfullscreen></iframe>  </td>
                <td>
                      <asp:button runat="server" Text="開始上課" class="btn" OnClick="C_click4"/>
                </td>
                     

            </tr>
        </table>
    </div>
        </div></form>
</asp:Content>