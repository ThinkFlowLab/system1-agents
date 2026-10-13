// S1A native desktop fixture: a tiny WinForms window with only noop/unlock/finish and a status label.
// Shown without activation and with WS_EX_NOACTIVATE so it never steals the user's focus.
// The app writes its own run-directory oracle: result.json (finished true/false) and app_events.jsonl.
// --id makes the process image name/title unique per run so only this run's window is ever targeted.
using System;
using System.Drawing;
using System.IO;
using System.Text;
using System.Windows.Forms;

namespace S1AFixture
{
    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            string mode = "normal";
            string id = "default";
            string outdir = ".";
            for (int i = 0; i + 1 < args.Length; i++)
            {
                if (args[i] == "--mode") mode = args[i + 1];
                if (args[i] == "--id") id = args[i + 1];
                if (args[i] == "--outdir") outdir = args[i + 1];
            }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            try
            {
                Application.Run(new FixtureForm(mode, id, outdir));
            }
            catch (Exception ex)
            {
                try { File.WriteAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "crash.txt"), ex.ToString()); } catch { }
                throw;
            }
        }
    }

    class FixtureForm : Form
    {
        readonly string _baseTitle;
        readonly string _mode;
        readonly bool _functional; // permanent: unlock and finish are inert, so the task has no solution
        readonly string _outdir;
        readonly Label _status;
        StreamWriter _events;
        bool _unlocked;
        bool _finished;

        public FixtureForm(string mode, string id, string outdir)
        {
            _baseTitle = "S1AFixture-" + id;
            _mode = mode;
            _functional = mode != "permanent";
            _outdir = outdir;

            Text = _baseTitle + " locked";
            Name = "S1AFixtureRoot";
            StartPosition = FormStartPosition.Manual;
            Location = new Point(40, 40);
            ClientSize = new Size(360, 150);
            ShowInTaskbar = false;
            FormBorderStyle = FormBorderStyle.FixedToolWindow;
            MaximizeBox = false;
            MinimizeBox = false;

            _status = new Label();
            _status.Name = "status";
            _status.AutoSize = true;
            _status.Location = new Point(20, 20);
            _status.Text = "locked";
            Controls.Add(_status);
            Controls.Add(MakeButton("noop", 20, 60, OnNoop));
            Controls.Add(MakeButton("unlock", 120, 60, OnUnlock));
            Controls.Add(MakeButton("finish", 220, 60, OnFinish));

            try
            {
                _events = new StreamWriter(Path.Combine(outdir, "app_events.jsonl"), true, Encoding.UTF8);
                _events.AutoFlush = true;
            }
            catch { _events = null; }
            WriteResult(false);
            AppendEvent("start");
        }

        Button MakeButton(string text, int x, int y, EventHandler handler)
        {
            Button button = new Button();
            button.Name = text;
            button.Text = text;
            button.Location = new Point(x, y);
            button.Size = new Size(90, 32);
            button.Click += handler;
            return button;
        }

        protected override bool ShowWithoutActivation { get { return true; } }

        protected override CreateParams CreateParams
        {
            get
            {
                CreateParams cp = base.CreateParams;
                cp.ExStyle |= 0x08000000; // WS_EX_NOACTIVATE
                cp.ExStyle |= 0x00000080; // WS_EX_TOOLWINDOW
                return cp;
            }
        }

        void AppendEvent(string kind)
        {
            if (_events == null) return;
            try
            {
                _events.WriteLine(
                    "{\"event\":\"" + kind + "\",\"unlocked\":" + (_unlocked ? "true" : "false") +
                    ",\"finished\":" + (_finished ? "true" : "false") +
                    ",\"pid\":" + System.Diagnostics.Process.GetCurrentProcess().Id + "}");
            }
            catch { }
        }

        void WriteResult(bool finished)
        {
            try
            {
                File.WriteAllText(
                    Path.Combine(_outdir, "result.json"),
                    "{\"finished\":" + (finished ? "true" : "false") + ",\"mode\":\"" + _mode +
                    "\",\"pid\":" + System.Diagnostics.Process.GetCurrentProcess().Id + "}");
            }
            catch { }
        }

        // The window title carries the state: the driver exposes it as a non-clickable TitleBar element,
        // so WindowEnv's progress and desktop.shows can read it (a WinForms Label is not in the tree).
        void SetStatus(string status)
        {
            _status.Text = status;
            Text = _baseTitle + " " + status;
        }

        void OnNoop(object sender, EventArgs e)
        {
            AppendEvent("noop"); // a true no-op: the window does not change
        }

        void OnUnlock(object sender, EventArgs e)
        {
            AppendEvent("unlock-clicked");
            if (!_functional || _unlocked) return;
            _unlocked = true;
            SetStatus("unlocked");
            AppendEvent("unlocked");
        }

        void OnFinish(object sender, EventArgs e)
        {
            AppendEvent("finish-clicked");
            if (!_functional || !_unlocked) return;
            _finished = true;
            SetStatus("finished OK");
            WriteResult(true);
            AppendEvent("finished");
        }
    }
}
