using System;
using System.Data.SqlClient;

public partial class Login : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {

    }

  

    protected void login_Click(object sender, EventArgs e)
    {
   
        SqlConnection sql = new SqlConnection(@"Data Source=SHINDA-LAPTOP01;Initial Catalog=LWeb;Integrated Security=True");

        string cmdsql = "Select count(*) from Member where Account=@account and password=@password";
        
        SqlCommand cmd = new SqlCommand(cmdsql, sql);
        cmd.Parameters.AddWithValue("@account", id_txt.Text);
        cmd.Parameters.AddWithValue("@password", password.Value);
       


        sql.Open();
        cmd.ExecuteNonQuery();
        int num = Convert.ToInt32(cmd.ExecuteScalar());
        if (num > 0)//改成大於0，表示此密碼在資料表中至少有一筆資料

        {
            string cmdsql2 = "Select count(*) from Member where Account=@account and password=@password and state='1'"; //學生

            SqlCommand cmd2 = new SqlCommand(cmdsql2, sql);
            cmd2.Parameters.AddWithValue("@account", id_txt.Text);
            cmd2.Parameters.AddWithValue("@password", password.Value);

            int num1 = Convert.ToInt32(cmd2.ExecuteScalar());
            if (num1 > 0)
            {
               Response.Redirect("Class.aspx");
            }
               else
                Response.Redirect("Class_teacher.aspx");




        }//End if

        else

        {
            msg.Text = "*帳號密碼錯誤*";
        }


    



    sql.Close();



       

    }

    
    protected void Register_Click(object sender, EventArgs e)
    {
        Response.Redirect("CreateUserWizard.aspx");

    }

    protected void Change_Click(object sender, EventArgs e)
    {
        Response.Redirect("ChangePassword.aspx");
    }
}