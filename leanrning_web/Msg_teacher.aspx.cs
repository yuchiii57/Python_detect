using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class Msg_teacher : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {

    }
    protected void GridView1_RowDataBound(object sender, GridViewRowEventArgs e)
    {

        if (e.Row.RowIndex != -1)
        {
            int id = e.Row.RowIndex + 1;
            e.Row.Cells[0].Text = id.ToString();
        }

    }

    protected void button1_Click(object sender, EventArgs e)
    {
        DateTime dt = DateTime.Now;
        string datetime = dt.ToString("yyyy-MM-dd HH:mm:ss");
        string cmdsql = "";
        if (label2.Text == "留言/提問:")
        {
            cmdsql = "INSERT INTO MsgBoard ([時間],[姓名/暱稱],[留言]) Values('" + datetime + "','" + name.Text + "','" + content.Text + "')";
        }
        else
        {
            cmdsql = "INSERT INTO MsgBoard ([時間],[姓名/暱稱],[留言]) Values('" + datetime + "','" + name.Text + "','" + label2.Text + content.Text + "')";
        }

        SqlConnection sql = new SqlConnection(@"Data Source=SHINDA-LAPTOP01;Initial Catalog=LWeb;Integrated Security=True");
        SqlCommand cmd = new SqlCommand(cmdsql, sql);
        sql.Open();
        cmd.ExecuteNonQuery();

        sql.Close();
        GridView1.DataBind();
        name.Text = string.Empty;
        content.Text = string.Empty;
        label2.Text = "留言/提問:";
    }

    protected void GridView1_RowDeleting(object sender, GridViewDeleteEventArgs e)
    {
        string sqlstr = "delete from MsgBoard where 時間='" + GridView1.DataKeys[e.RowIndex].Value.ToString() + "'";
        SqlConnection sql = new SqlConnection(@"Data Source=SHINDA-LAPTOP01;Initial Catalog=LWeb;Integrated Security=True");
        SqlCommand cmd = new SqlCommand(sqlstr, sql);
        sql.Open();
        cmd.ExecuteNonQuery();

        sql.Close();
        GridView1.DataBind();
    }
    protected void GridView1_RowPicking(object sender, GridViewSelectEventArgs e)
    {
        try
        {
            DateTime dt = DateTime.Now;
            string datetime = dt.ToString("yyyy-MM-dd HH:mm:ss");
            label2.Text = "***我要回覆" + GridView1.SelectedRow.Cells[0].Text + "樓:";
        }
        catch (Exception ex)
        {
            Response.Write(ex);
        }


    }
}