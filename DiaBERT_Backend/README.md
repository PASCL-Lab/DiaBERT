Hosting on Fly.io
The very first step:
Download and install Fly.io’s tool (called Flyctl) and log in.
Why?
Flyctl is the “remote control” for Fly.io. Without it, your computer doesn’t know how to talk to Fly’s servers.

How to do it (on Windows, simplest way):
1.	Go to this page: Fly.io Download.
2.	Download the Windows ZIP file.
3.	Unzip it (for example into a folder like C:\flyctl\).
4.	Open that folder and double-click flyctl.exe to make sure it runs.
o	Or in Command Prompt:
o	C:\flyctl\flyctl.exe version
(It should show you the version number.)
5.	Then, log in to your Fly.io account:
6.	C:\flyctl\flyctl.exe auth login
→ This will open a browser window. Just click “Authorize” to connect your account.


Step 2: Go into your project folder (the folder with your app)

Why?
Fly.io needs to know what you want to send online. That “what” is your project folder — the place where your code (app.py), your model files, and your instructions (Dockerfile or requirements.txt) live.

How (Windows, Command Prompt):
1.	Open Command Prompt (the black window you type in).
2.	Type cd (which means “change directory”) followed by the path to your project. For example:
cd "C:\Users\linda\OneDrive\Desktop\DiaBERT_Backend"
3.	Press Enter.
o	If it works, the little text on the left of the window will change to show you’re inside the DiaBERT_Backend folder.


What should be inside the backend folder (In my case)
When you go into
C:\Users\linda\OneDrive\Desktop\DiaBERT_Backend
you should see at least these items:
1.	app.py
o	This is your main server file (Flask app).
o	It has all your routes (/ping, /predict, etc.).
2.	Model files (the “brain” of your app)
o	newbiobert_finetuned_3class.onnx
o	combined_embeddings.pt
o	combined_texts.pt
o	newbiobert_model_3class/newbiobert_model_3class/pytorch_model.bin
These are loaded by app.py when someone sends a request.
3.	requirements.txt (or just requirements)
o	A list of all the Python packages your app needs (flask, torch, onnxruntime, etc.).
o	Fly will read this to know what to install in the cloud.
4.	Dockerfile (for your setup)
o	Instructions Fly uses to package your app.
o	Since you already have this, Fly is building with Docker (confirmed by the logs).
5.	fly.toml (created by fly launch)
o	The config file Fly.io uses to know how to run your app.
o	This file is generated automatically, then we tweak it.
6.	(Optional) .dockerignore
o	Tells Docker what files to skip when building (e.g., don’t copy .git or large temp files).
o	I created this when  I was dealing with Git LFS and large model files — but its role is just to keep the Docker image lean, not to fetch LFS data.

Step 3: Tell Fly.io to create an app for you

Why?
Think of this like opening an account for your project. Fly.io needs to reserve a name and a place in the cloud where your app will live.

How (still in Command Prompt, inside your project folder):
Type this command and press Enter:
C:\flyctl\flyctl.exe launch --no-deploy

What happens next:
1.	Fly.io will ask you a few simple questions:
o	App name → choose something unique (like diabert-app-123).
o	Region → pick a place near you (for Canada, usually yyz for Toronto).
o	Database → when it asks about Postgres, just say No (you don’t need it).
2.	When you’re done, Fly.io creates a small file called fly.toml in your folder.
o	This file is like a recipe card that remembers how your app should run.

If your backend doesn’t use OpenAI at all, then Step 4 is optional — you can skip it.
Step 4: Give your app the secret key it needs
Why?
Your backend uses OpenAI to create explanations. For that to work, the app needs your OpenAI API key.
Instead of writing it in your code (unsafe), we give it to Fly.io as a secret. Fly stores it safely and passes it to your app when it runs.
How (in Command Prompt, still inside your project folder):
Type this command, but replace the xxxxx with your real key:
C:\flyctl\flyctl.exe secrets set OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
What happens:
•	Fly.io saves your key.
•	Your app will now “see” it when it starts running.
•	You don’t have to change your code — it already checks for OPENAI_API_KEY in the environment.


Step 5: Send your app to Fly.io (the deploy step)
Why?
Up to now, you’ve only prepared everything (told Fly about your app, added your secret key, etc.).
Now it’s time to actually upload your backend and models to Fly.io so the app can run online.

How (still in Command Prompt, inside your project folder):
C:\flyctl\flyctl.exe deploy
What happens:
1.	Fly.io looks at your folder and finds the Dockerfile.
2.	It copies all your files (your code + models).
3.	It builds an image (kind of like a “package”) with Python, your libraries, and your app.
4.	It sends that image to Fly’s servers.
5.	Fly starts a virtual machine with your app running inside it.
________________________________________
How you’ll know it worked:
•	You’ll see logs with lines like:
•	==> Building image with Docker
•	...
•	image size: 5.3 GB
•	==> Monitoring deployment
•	✔ App is up and running
•	At the end, Fly will give you a URL such as:
•	https://diabert.fly.dev


