<%@ Page Language="C#" AutoEventWireup="true" CodeFile="Msg_teacher.aspx.cs" Inherits="Msg_teacher" MasterPageFile="~/MasterPage2.master" Title="MessageBoard" %>

<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder2" Runat="Server">
     <link href="../css/Style.css" type="text/css" rel="stylesheet" />
     <form id="form1" runat="server">
         <asp:GridView ID="GridView1" runat="server" AutoGenerateColumns="False" DataKeyNames="時間" DataSourceID="SqlDataSource1"  CssClass="gvStyle" AlternatingRowStyle-CssClass="alt" PagerStyle-CssClass="pgr" style="margin-left:100px" OnRowDataBound="GridView1_RowDataBound" OnSelectedIndexChanging="GridView1_RowPicking">
<AlternatingRowStyle CssClass="alt"></AlternatingRowStyle>
             <Columns>
                 <asp:BoundField DataField="時間" HeaderText="No" ReadOnly="True" SortExpression="時間" />
                 <asp:BoundField DataField="姓名/暱稱" HeaderText="姓名/暱稱" SortExpression="姓名/暱稱" />
                 <asp:BoundField DataField="留言" HeaderText="留言" SortExpression="留言" />
                  <asp:CommandField HeaderText="回覆" ShowSelectButton="True" />
                   <asp:CommandField HeaderText="刪除" ShowDeleteButton="True" />
             </Columns>

<PagerStyle CssClass="pgr"></PagerStyle>
         </asp:GridView>
         <asp:SqlDataSource ID="SqlDataSource1" runat="server" ConnectionString="<%$ ConnectionStrings:LWebConnectionString %>" SelectCommand="SELECT * FROM MsgBoard " DeleteCommand="DELETE FROM MsgBoard WHERE (時間 = @時間)">
             <DeleteParameters>
                 <asp:Parameter Name="時間" />
             </DeleteParameters>
         </asp:SqlDataSource>
    <div style="margin-left:100px;margin-top:30px;margin-bottom:30px">
    <asp:Label ID="label1" runat="server">我是:</asp:Label>
         <asp:TextBox ID="name" runat="server"></asp:TextBox><br />
         <asp:Label ID="label2" runat="server" >留言/提問:</asp:Label>
         <asp:TextBox ID="content" runat="server" TextMode="MultiLine" style="margin-top:30px"></asp:TextBox><br />

         <asp:Button ID="button1" runat="server" Text="留言" style="margin-left:150px" OnClick="button1_Click" />
</div>
     </form>



</asp:Content>