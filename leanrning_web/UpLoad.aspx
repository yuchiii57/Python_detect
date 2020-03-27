<%@ Page Language="C#" Title="FileUpload" AutoEventWireup="true" CodeFile="UpLoad.aspx.cs"  Inherits="UpLoad" %>

  
 
<!DOCTYPE html>

<link href="vendors/css/flag-icon.min.css" rel="stylesheet"/>
  <link href="vendors/css/font-awesome.min.css" rel="stylesheet"/>
  <link href="vendors/css/simple-line-icons.min.css" rel="stylesheet"/>
  <link href="css/style1.css" rel="stylesheet"/>
    <link href="../css/Style.css" type="text/css" rel="stylesheet" />


<body class="app header-fixed sidebar-fixed aside-menu-fixed aside-menu-hidden">
    
    <header class="app-header navbar">
        <button class="navbar-toggler mobile-sidebar-toggler d-lg-none" type="button">
            <span class="navbar-toggler-icon"></span>
        </button>
        <a class="navbar-brand" href="#"  style="border-right:solid 1px #fff">
            <img style="height:40px;float:left " src="imgs/logo.png" />
            <span style="color:#fff;height:40px;padding-left:10px"><b>VAI</b></span>
          
          
        </a>
        <button class="navbar-toggler sidebar-toggler d-md-down-none h-100 b-r-1 px-3" type="button">
            <span class="navbar-toggler-icon"></span>
        </button>
        <ul class="nav navbar-nav d-md-down-none mr-auto">
            <form class="form-inline px-4">
                <i class="fa fa-search"></i>
                <input class="form-control" type="text" placeholder="Search...">
            </form>
        </ul>
        <ul class="nav navbar-nav ml-auto">
            
          
              <li>
                 <a href="Login.aspx"> <img src="imgs/logout.png" /></a> 
              </li>
            </ul>
</header>
    
    <div style="margin-left:100px">
         <asp:FileUpload ID="FileUpload1" runat="server" /> 
        <asp:Label ID="label1" runat="server"></asp:Label>
        <asp:Button ID="button1" runat="server" OnClick="button1_Click" Text="上傳" />
    
    </div>
    </body>
