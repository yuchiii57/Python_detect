<%@ Page Language="C#" AutoEventWireup="true" CodeFile="CreateUserWizard.aspx.cs" Inherits="CreateUserWizard" %>

<!DOCTYPE html>

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <title>Login</title>
      <style type="text/css">
     a{
         color:#fff;
     } 
   
          </style>
    <link href="../css/Style.css" type="text/css" rel="stylesheet"/>
</head>
<body style="font-family:Calibri;background-color:#fff " >

    <form id="form1" runat="server">
   <div style="color:rgb(127,127,127);text-align:center;margin-top:50px;font-size:55px;">
        <img src="imgs/logo.png" />
       VISUAL- ATTENTION- IDENTIFICATION
      
   </div>
  
        <div style="margin-top:50px;" >
           
            <div class="center" >
               <h2 class="text-center" style="color:#808080"> <b>註冊新使用者</b> </h2>
                <asp:Label ID="Label1" runat="server">姓名:</asp:Label>
                <asp:TextBox ID="name" runat="server" style="width:200px;"></asp:TextBox>
                 <asp:Label ID="name2" runat="server" style="color:#ff0000"></asp:Label>

                <asp:Dropdownlist ID="state" runat="server" >
                    <asp:ListItem Value="0" Text="教師"></asp:ListItem>
                     <asp:ListItem Value="1" Text="學生"></asp:ListItem>
                </asp:Dropdownlist><br />
                <asp:Label ID="ac" runat="server">帳號:</asp:Label>
                <asp:TextBox ID="account" runat="server" style="width:200px;"></asp:TextBox>
                <asp:Label ID="account2" runat="server" style="color:#ff0000"></asp:Label><br />

                <asp:Label ID="pass" runat="server">密碼:</asp:Label>
                 <input type="password" id="password" name="YourPassword"  runat="server" style="width:200px;" />
               <asp:Label ID="password2" runat="server" style="color:#ff0000"></asp:Label><br />

                <asp:Button ID="enter" runat="server" OnClick="enter_Click" Text="確認" style="margin-left:80px" /> 
                <asp:Button ID="cancel" runat="server" OnClick="cancel_Click"  Text="取消" style="margin-left:10px" />

             
                </div>
   </div>     
    </form>
</body>
</html>
