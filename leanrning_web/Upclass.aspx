<%@ Page Language="C#" AutoEventWireup="true" MasterPageFile="~/MasterPage2.master"  Title="Class" CodeFile="Upclass.aspx.cs" Inherits="Upclass" %>

<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder2" Runat="Server">
      
          <form id="form1" runat="server">
              <div style="text-align:center;margin-left:20px;margin-right:20px;"> 
          <asp:GridView ID="GridView1" runat="server" AutoGenerateColumns="False" DataKeyNames="id" DataSourceID="SqlDataSource1" CssClass="gvStyle" AlternatingRowStyle-CssClass="alt" PagerStyle-CssClass="pgr" style="margin-left:100px">
<AlternatingRowStyle CssClass="alt"></AlternatingRowStyle>
              <Columns>
                  <asp:BoundField DataField="FileName" HeaderText="FileName" SortExpression="FileName" />
                  <asp:BoundField DataField="FilePath" HeaderText="FilePath" SortExpression="FilePath" />
                   <asp:CommandField HeaderText="刪除" ShowDeleteButton="True" />
              </Columns>

<PagerStyle CssClass="pgr"></PagerStyle>
          </asp:GridView>
          <asp:SqlDataSource ID="SqlDataSource1" runat="server" ConnectionString="<%$ ConnectionStrings:LWebConnectionString %>" SelectCommand="SELECT [FileName], [FilePath], [id] FROM [File]" DeleteCommand="DELETE from [File] WHERE ID=@ID">
              <DeleteParameters>
                  <asp:Parameter Name="ID" />
              </DeleteParameters>
                  </asp:SqlDataSource>
      <link href="../css/Style.css" type="text/css" rel="stylesheet" />
   
        
        

        </div>
      </form>
    </asp:Content>