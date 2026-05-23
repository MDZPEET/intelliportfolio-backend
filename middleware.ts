import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';

// ✅ เพิ่ม /login, /register และ /admin เข้าไปในรายชื่อ "หน้าสาธารณะ"
const isPublicRoute = createRouteMatcher([
  '/', 
  '/login(.*)',    
  '/register(.*)',
  '/admin(.*)'
]);

export default clerkMiddleware(async (auth, req) => {
  if (!isPublicRoute(req)) {
    await auth.protect(); 
  }
});

export const config = {
  matcher: [
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    '/(api|trpc)(.*)',
  ],
};