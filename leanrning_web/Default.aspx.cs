using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Linq;
using System.ServiceModel.Channels;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class _Default : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {

    }
    protected void button2_Click(object sender, EventArgs e)
    {

        Response.Redirect("Class_teacher.aspx");

    }

    protected void button1_Click(object sender, EventArgs e)
    {
        string saveDir = @"\class\";
        //因為反斜線會被解析,使用@方便閱讀
        //還是可寫成String saveDir= "\\FileUpload\\"; 
        string appPath = Request.PhysicalApplicationPath;//取得目錄完整位址

        if (FileUpload1.HasFile)
        {
            string savePath = appPath + saveDir +
                Server.HtmlEncode(FileUpload1.FileName);
            FileUpload1.SaveAs(savePath);
            label1.Text = "Your file was uploaded successfully.";
        }
        else
        {
            label1.Text = "You did not specify a file to upload.";
        }
        

        
        if (FileUpload1.HasFile)
        {
            string MessageID ="1";
            SubmitFile(MessageID, FileUpload1.PostedFile);
            label2.Text += "檔案上傳完成!";
            
        }
      



    }

    protected void SubmitFile(string MessageID, IList<HttpPostedFile> FileList)//上傳檔案及回傳檔案清單
    {
        using (SqlConnection nowConnection = new SqlConnection(@"Data Source=SHINDA-LAPTOP01;Initial Catalog=LWeb;Integrated Security=True"))
        {
            nowConnection.Open();//開啟連線
            foreach (HttpPostedFile postedFile in FileList)
            {
                String saveDir = @"class\";
                //String saveDir= "\\FileUpload\\";
                String appPath = Request.PhysicalApplicationPath;//根目錄之實體路徑
                String fileName, checkPath;
                fileName = postedFile.FileName;
                string tempfileName = fileName;
                checkPath = appPath + saveDir + fileName;
                if (System.IO.File.Exists(checkPath))//避免檔案重複儲存
                {
                    int counter = 2;
                    while (System.IO.File.Exists(checkPath))
                    {
                        tempfileName = "(" + counter.ToString() + ")" + fileName;
                        checkPath = appPath + saveDir + tempfileName;
                        counter = counter + 1;
                    }
                    fileName = tempfileName;
                }
                string filePathName = appPath + saveDir + tempfileName;
                postedFile.SaveAs(filePathName);
                using (SqlCommand command = new SqlCommand())
                {
                    DateTime dt = DateTime.Now;
                    string d=dt.ToString("MMddHHmmss");
                    int num = Convert.ToInt32(d.ToString());

                    command.Parameters.Clear();//清空參數
                    command.Parameters.Add("@ID", SqlDbType.Int).Value = MessageID;
                    command.Parameters.AddWithValue("@FILENAME", postedFile.FileName);
                    command.Parameters.AddWithValue("@FILEPATH", filePathName);
                    command.CommandText = @"insert into [File] (ID,MessageID,FileName,FilePath)
                                        values('"+num+"',@ID,@FILENAME,@FILEPATH)";
                    command.Connection = nowConnection;//資料庫連接
                    command.ExecuteNonQuery();
                }
            }

        }

    }
}