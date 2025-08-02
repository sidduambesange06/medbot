// static/js/supabase.js

// Make sure you have included the Supabase JS library before this script
// Example: <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js"></script>

const { createClient } = supabase;

// Use the correct Supabase project URL (not the /auth/v1/callback endpoint)
const supabaseUrl = 'https://vyzzvdimsuaeknpmyggt.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZ5enp2ZGltc3VhZWtucG15Z2d0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM3NzM5NzQsImV4cCI6MjA2OTM0OTk3NH0.BPnPM0fdomDXm1qVwzPlvlW4sW-WazLByAF0X8m1u94';

const supabaseClient = createClient(supabaseUrl, supabaseKey);

window.supabase = supabaseClient;
