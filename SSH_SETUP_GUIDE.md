# SSH Key Setup Guide for GitHub Access

## 🚀 Quick Start

**Why SSH?** SSH keys provide secure, password-free access to GitHub. Once set up, you won't need to enter your password every time you push code.

## 📋 Prerequisites

- GitHub account
- Git installed on your computer
- Terminal/Command Prompt access

## 🔑 Step-by-Step Setup

### Step 1: Check for Existing SSH Keys

**Windows (PowerShell/Command Prompt):**
```bash
ls ~/.ssh
```

**macOS/Linux (Terminal):**
```bash
ls ~/.ssh
```

**If you see files like `id_rsa`, `id_ed25519`, etc., you already have SSH keys!**

### Step 2: Generate New SSH Key

**Windows (PowerShell/Command Prompt):**
```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
```

**macOS/Linux (Terminal):**
```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
```

**When prompted:**
- **File location**: Press `Enter` to accept default
- **Passphrase**: Enter a secure passphrase (recommended) or press `Enter` for no passphrase

### Step 3: Start SSH Agent

**Windows (PowerShell):**
```bash
Start-Service ssh-agent
ssh-add ~/.ssh/id_ed25519
```

**macOS/Linux (Terminal):**
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### Step 4: Copy Your Public Key

**Windows (PowerShell):**
```bash
Get-Content ~/.ssh/id_ed25519.pub | Set-Clipboard
```

**macOS (Terminal):**
```bash
pbcopy < ~/.ssh/id_ed25519.pub
```

**Linux (Terminal):**
```bash
cat ~/.ssh/id_ed25519.pub
# Manually copy the output
```

### Step 5: Add SSH Key to GitHub

1. **Go to GitHub.com** and sign in
2. **Click your profile picture** → **Settings**
3. **Click "SSH and GPG keys"** in the left sidebar
4. **Click "New SSH key"**
5. **Title**: Enter a descriptive name (e.g., "My Laptop")
6. **Key**: Paste your public key (from Step 4)
7. **Click "Add SSH key"**

### Step 6: Test Your Connection

```bash
ssh -T git@github.com
```

**You should see:**
```
Hi username! You've successfully authenticated, but GitHub does not provide shell access.
```

## 🔧 Troubleshooting

### "Permission denied" Error
```bash
# Check if your key is loaded
ssh-add -l

# If empty, add your key
ssh-add ~/.ssh/id_ed25519
```

### "Host key verification failed"
```bash
# Remove old GitHub keys
ssh-keygen -R github.com

# Try connecting again
ssh -T git@github.com
```

### Windows Issues
- **Use PowerShell** instead of Command Prompt
- **Run as Administrator** if needed
- **Enable OpenSSH** in Windows Features

## 📚 Helpful Resources

### Official Documentation
- **[GitHub SSH Setup Guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)**
- **[GitHub SSH Troubleshooting](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/troubleshooting-ssh)**

### Video Tutorials
- **[GitHub SSH Key Setup (Windows)](https://www.youtube.com/watch?v=8X4u9sca3Io)**
- **[GitHub SSH Key Setup (macOS)](https://www.youtube.com/watch?v=Wm6aIUjdr0Q)**
- **[GitHub SSH Key Setup (Linux)](https://www.youtube.com/watch?v=H5qNpRGB7Qw)**

### Interactive Tutorials
- **[GitHub Learning Lab: SSH](https://lab.github.com/githubtraining/introduction-to-github)**
- **[GitHub Skills: SSH](https://skills.github.com/)**

## 🎯 For Our Project

### Once SSH is Set Up

1. **Clone the repository:**
```bash
git clone git@github.com:NERD-Community-Ethiopia/generative-ai-course.git
cd generative-ai-course
```

2. **Create your branch:**
```bash
git checkout -b intern/your-name
git push -u origin intern/your-name
```

3. **Start developing:**
```bash
make install
make setup
```

## ❓ Need Help?

### Common Issues & Solutions

**"ssh-keygen not found"**
- **Windows**: Install Git for Windows or enable OpenSSH
- **macOS**: Install Xcode Command Line Tools
- **Linux**: Install openssh-client

**"Permission denied (publickey)"**
- Check if your key is added to GitHub
- Verify your SSH agent is running
- Ensure you're using the correct email

**"Host key verification failed"**
- Remove old GitHub keys and try again
- Check your internet connection

### Getting Support

1. **Check the troubleshooting section above**
2. **Search GitHub's documentation**
3. **Ask in our group chat**
4. **Create a GitHub issue** (if it's project-related)

## ✅ Verification Checklist

- [ ] SSH key generated successfully
- [ ] SSH key added to GitHub
- [ ] SSH connection test passes
- [ ] Can clone repository with SSH
- [ ] Can push to your branch

## 🎉 Success!

Once you complete this setup, you'll have:
- ✅ Secure, password-free GitHub access
- ✅ Faster git operations
- ✅ Better security practices
- ✅ Professional development workflow

---

**Remember**: SSH keys are like digital passports for your computer. Keep them secure and never share your private key! 