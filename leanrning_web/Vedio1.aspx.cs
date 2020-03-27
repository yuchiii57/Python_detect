using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class Vedio1 : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {

    }
    protected void data_click(object sender, EventArgs e)
    {
        DateTime dd = DateTime.Now;
        string dt = dd.ToString();
        FileStream fs = File.Create(@"C:\testword\Vedio1_Log.txt");
        StreamWriter sw = new StreamWriter(fs);
        sw.Write(showbox.Value);
        sw.Flush();
        sw.Close();
        fs.Close();
        showbox.Value = string.Empty;
    }
}