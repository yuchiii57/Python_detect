<%@ Page Language="C#" AutoEventWireup="true" CodeFile="Default.aspx.cs" Inherits="_Default" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title></title>
</head>
<body>
    <form id="form1" runat="server">
    <div>
    
        <asp:FileUpload ID="FileUpload1" runat="server" /> 
        <asp:Button ID="button1" runat="server" OnClick="button1_Click" Text="上傳" />
        <asp:Label ID="label1" runat="server"></asp:Label>
        <asp:Label ID="label2" runat="server"></asp:Label>
       <asp:Button ID="button2" runat="server" OnClick="button2_Click" Text="不再上傳" />
    </div>
    </form>
</body>
</html>
