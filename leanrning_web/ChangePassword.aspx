<%@ Page Language="C#" AutoEventWireup="true" CodeFile="ChangePassword.aspx.cs" Inherits="ChangePassword" %>

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

            <div class="center">
       
        <asp:ChangePassword ID="ChangePassword1" runat="server" BackColor="White" BorderColor="" BorderPadding="5" BorderStyle="Solid" BorderWidth="1px" ContinueDestinationPageUrl="~/Login.aspx" DisplayUserName="True" Font-Names="Verdana" Font-Size="15px" Height="100%" Width="400px" CancelDestinationPageUrl="~/Login.aspx" ChangePasswordTitleText="Change Your Password" EnableViewState="False">
            <CancelButtonStyle BackColor="White" BorderColor="" BorderStyle="Solid" BorderWidth="1px" Font-Names="Verdana" Font-Size="13px" ForeColor=""  />
            <ChangePasswordButtonStyle BackColor="White" BorderColor="" BorderStyle="Solid" BorderWidth="1px" Font-Names="Verdana" Font-Size="13px" ForeColor=""  />
            <ContinueButtonStyle BackColor="White" BorderColor="" BorderStyle="Solid" BorderWidth="1px" Font-Names="Verdana" Font-Size="13px" ForeColor="" Width="100px"/>
            <InstructionTextStyle Font-Italic="True" ForeColor="#7f7f7f" />
            <PasswordHintStyle Font-Italic="True" ForeColor="" />
            <TextBoxStyle Font-Size="15px" />
            <TitleTextStyle BackColor="" Font-Bold="True" Font-Size="20px" ForeColor="#7f7f7f" />
           <LabelStyle ForeColor="#7f7f7f" />
            <ValidatorTextStyle ForeColor="Red"/>
             </asp:ChangePassword>
                </div>
   </div>     
    </form>
</body>
</html>
