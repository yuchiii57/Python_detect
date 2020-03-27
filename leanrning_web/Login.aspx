<%@ Page Language="C#" AutoEventWireup="true" CodeFile="Login.aspx.cs" Inherits="Login" %>



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

   <div style="color:rgb(127,127,127);text-align:center;margin-top:50px;font-size:60px;">
         <img src="imgs/logo.png" />
       VISUAL- ATTENTION- IDENTIFICATION
   </div>
  
        <div style="margin-top:50px;" >

                <div class="left">
       <img src="imgs/app_development.gif" style="width:600px" />
   </div>
            <div class="right">
       <table style="line-height:60px; color:rgb(127,127,127)">
           <tr>
               <td></td>
            <td style="font-size:40px;text-align:left;color:cornflowerblue">
                      
                    Login
                  </td>
                     </tr>
           <tr>
               <td class="login_t"> ID:</td>
         
           <td>
                 <asp:TextBox ID="id_txt" runat="server" style="width:200px;border-radius:4px;"></asp:TextBox>
           </td>
            </tr>
       <tr>
               <td class="login_t">  Password: </td>
         
           <td>
            <input type="password" id="password" name="YourPassword"  runat="server" style="width:200px;border-radius:4px;" />
           </td>
            </tr>
           <tr>
               <td></td>
               <td>
                   <asp:Label ID="msg" runat="server" style="color:red"></asp:Label>
               </td>
           </tr>
           <tr>
               <td></td>
               <td class="signbtn">
                  
              <asp:Button runat="server" ID="login" OnClick="login_Click" Text="登入"  />
                  
              <asp:Button runat="server" ID="Register"  Text="註冊 " OnClick="Register_Click" style="margin-left:10px"/> 
                   
             
               </td>
           </tr>

       </table>
   
   </div>     
 
        </div>
 
     </form>
</body>

