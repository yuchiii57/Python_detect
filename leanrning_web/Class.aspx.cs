using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class Class1 : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {

    }
    protected void C_click(object sender, EventArgs e)
    {
        DateTime dt = DateTime.Now;
        string datetime = dt.ToString("yyyy-MM-dd HH:mm:ss");
        SqlConnection sql = new SqlConnection(@"Data Source=SHINDA-LAPTOP01;Initial Catalog=LWeb;Integrated Security=True");
        string cmdsql = "INSERT INTO Class([time],[v_name],[watch]) VALUES ('" + datetime + "','How to build (and rebuild) trust | Frances Frei','v') ";
        SqlCommand cmd = new SqlCommand(cmdsql, sql);
        sql.Open();
        cmd.ExecuteNonQuery();

        sql.Close();
        Response.Redirect("Vedio1.aspx");
    }
    protected void C_click2(object sender, EventArgs e)
    {
        DateTime dt = DateTime.Now;
        string datetime = dt.ToString("yyyy-MM-dd HH:mm:ss");
        SqlConnection sql = new SqlConnection(@"Data Source=SHINDA-LAPTOP01;Initial Catalog=LWeb;Integrated Security=True");
        string cmdsql = "INSERT INTO Class([time],[v_name],[watch]) VALUES ('" + datetime + "','The person you really need to marry | Tracy McMillan','v') ";
        SqlCommand cmd = new SqlCommand(cmdsql, sql);
        sql.Open();
        cmd.ExecuteNonQuery();

        sql.Close();
        Response.Redirect("Vedio1.aspx");
    }
    protected void C_click3(object sender, EventArgs e)
    {
        DateTime dt = DateTime.Now;
        string datetime = dt.ToString("yyyy-MM-dd HH:mm:ss");
        SqlConnection sql = new SqlConnection(@"Data Source=SHINDA-LAPTOP01;Initial Catalog=LWeb;Integrated Security=True");
        string cmdsql = "INSERT INTO Class([time],[v_name],[watch]) VALUES ('" + datetime + "','The next step in nanotechnology | George Tulevski','v') ";
        SqlCommand cmd = new SqlCommand(cmdsql, sql);
        sql.Open();
        cmd.ExecuteNonQuery();

        sql.Close();
        Response.Redirect("Vedio1.aspx");
    }
    protected void C_click4(object sender, EventArgs e)
    {
        DateTime dt = DateTime.Now;
        string datetime = dt.ToString("yyyy-MM-dd HH:mm:ss");
        SqlConnection sql = new SqlConnection(@"Data Source=SHINDA-LAPTOP01;Initial Catalog=LWeb;Integrated Security=True");
        string cmdsql = "INSERT INTO Class([time],[v_name],[watch]) VALUES ('" + datetime + "','微笑隱藏的力量 | Ron Gutman','v') ";
        SqlCommand cmd = new SqlCommand(cmdsql, sql);
        sql.Open();
        cmd.ExecuteNonQuery();

        sql.Close();
        Response.Redirect("Vedio1.aspx");
    }
}