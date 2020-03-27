<%@ Page Language="C#" AutoEventWireup="true" CodeFile="Class_Record.aspx.cs" MasterPageFile="~/MasterPage.master" Inherits="Class_Record" Title="Class Record"%>


<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder2" Runat="Server">
    <form runat="server" id="form1">
         <asp:GridView ID="GridView1" runat="server" AutoGenerateColumns="False" DataKeyNames="time" DataSourceID="SqlDataSource1" CssClass="gvStyle" AlternatingRowStyle-CssClass="alt" PagerStyle-CssClass="pgr" style="margin-left:100px" >
<AlternatingRowStyle CssClass="alt"></AlternatingRowStyle>
             <Columns>
                  <asp:BoundField DataField="time" HeaderText="時間" ReadOnly="True" SortExpression="time" />
                 <asp:BoundField DataField="v_name" HeaderText="課程" SortExpression="v_name" />
                 <asp:BoundField DataField="watch" HeaderText="已看過" SortExpression="watch" />
                
             </Columns>

<PagerStyle CssClass="pgr"></PagerStyle>
         </asp:GridView>
         <asp:SqlDataSource ID="SqlDataSource1" runat="server" ConnectionString="<%$ ConnectionStrings:LWebConnectionString %>" SelectCommand="SELECT * FROM [Class]"></asp:SqlDataSource>
      <link href="../css/Style.css" type="text/css" rel="stylesheet" />
         <div style="text-align:center;margin-left:20px;margin-right:20px;">
   
             </div>

</form>
    </asp:Content>