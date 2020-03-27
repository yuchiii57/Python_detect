using System;
using System.Data;
using System.Data.SqlClient;      //加入這一個命名空間
using System.Data.SQLite;

public partial class Default2 : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {

        //string sql = "SELECT * FROM userInfo";
        ////string conStr = "D:/sqlliteDb/document.db";
        //string connStr = @"Data Source=" + @"C:\testword\Sqlite01.sqlite;Initial Catalog=sqlite;Integrated Security=True;Max Pool Size=10";
        //using (SQLiteConnection conn = new SQLiteConnection(connStr))
        //{
        //    //conn.Open();
        //    using (SQLiteDataAdapter ap = new SQLiteDataAdapter(sql, conn))
        //    {
        //        DataSet ds = new DataSet();
        //        ap.Fill(ds);

        //        DataTable dt = ds.Tables[0];
        //    }
      
        SqlConnection sql = new SqlConnection(@"Data Source=SHINDA-LAPTOP01;Initial Catalog=LWeb;Integrated Security=True");
        string cmdsql = "SELECT * FROM Table_1";
        SqlCommand cmd = new SqlCommand(cmdsql, sql);
        sql.Open();
        cmd.ExecuteNonQuery();

        sql.Close();
    }
}
    
    

        //}
    



   
    
