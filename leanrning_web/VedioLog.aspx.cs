using System;
using System.IO;

public partial class _Default : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {
        showbox.InnerText = "";
    }
      public void Write(string path)
          {
       
             FileStream fs = new FileStream(path, FileMode.Create);
            StreamWriter sw = new StreamWriter(fs);
        //開始寫入
        DateTime dt = DateTime.Now;
            sw.Write(dt.ToString()+showbox.InnerHtml);
           //清空緩衝區
           sw.Flush();
           //關閉流
            sw.Close();
            fs.Close();
 }


    protected void btn_Click(object sender, EventArgs e)
    {
        //Write("C:/testword/log.txt");
    }
}
