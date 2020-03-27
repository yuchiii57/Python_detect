using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class UpLoad : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {

    }

    protected void button1_Click(object sender, EventArgs e)
    {
        if (FileUpload1.HasFile)
        {
            String fileName = FileUpload1.FileName;  //-- User上傳的檔名（不包含 Client端的路徑！）

        

            //*** 方法二 *****
            String savePath = Server.MapPath("~/C:/Web/");
          

            String saveResult = savePath + fileName;
            //-- 重點！！必須包含 Server端的「目錄」與「檔名」，才能使用 .SaveAs()方法！
            FileUpload1.SaveAs(saveResult);

            label1.Text = "<b>上傳成功</b>，檔名---- " + fileName;
            label1.Text += "<br />路徑檔名---- " + savePath;
        }
        else
        {
            label1.Text = "請先挑選檔案之後，再來上傳";
        }
        label1.Text = string.Empty;
    }
}