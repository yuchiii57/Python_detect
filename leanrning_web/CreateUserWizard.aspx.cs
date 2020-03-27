using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

public partial class CreateUserWizard : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {

    }



    protected void enter_Click(object sender, EventArgs e)
    {
        if (name.Text == "")
            name2.Text = "*必填";
        if (account.Text == "")
            account2.Text = "*必填";
        if (password.Value == "")
            password2.Text = "*必填";
       if(name.Text!="" && account.Text != "" && password.Value != "")
        {

            SqlConnection sql1 = new SqlConnection(@"Data Source=SHINDA-LAPTOP01;Initial Catalog=LWeb;Integrated Security=True");

            string cmdsql1 = "Select count(*) from Member where Account=@account";

            SqlCommand cmd1 = new SqlCommand(cmdsql1, sql1);
            cmd1.Parameters.AddWithValue("@account", account.Text);

            sql1.Open();
            cmd1.ExecuteNonQuery();
            int num = Convert.ToInt32(cmd1.ExecuteScalar());
            sql1.Close();
            if (num > 0)//改成大於0，表示此密碼在資料表中至少有一筆資料

            {
                account2.Text = "*已有此帳號";
            }

            else {
                SqlConnection sql = new SqlConnection(@"Data Source=SHINDA-LAPTOP01;Initial Catalog=LWeb;Integrated Security=True");
         string cmdsql = " INSERT INTO Member ([Account],[Name],[Password],[state]) Values('" + account.Text + "','" + name.Text + "','" + password.Value.ToString() + "','"+state.SelectedValue+"')";
        SqlCommand cmd = new SqlCommand(cmdsql, sql);
        sql.Open();
        cmd.ExecuteNonQuery();

        sql.Close();

            Response.Write("<script language=javascript>alert('註冊成功，請按[取消]重新登入!');</" + "script>");
            name.Text = string.Empty;
            account.Text = string.Empty;
 }



        }
       
    }

    protected void cancel_Click(object sender, EventArgs e)
    {
        Response.Redirect("Login.aspx");
    }
}